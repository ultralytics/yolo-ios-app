// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Accelerate
import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for monocular depth models that produce one metric distance per pixel.
public final class DepthEstimator: BasePredictor, @unchecked Sendable {
  /// Jet colormap, channel = clamp(1.5 - |4t - center|, 0, 1) at t = index / 255; entry 0 is black (invalid pixels).
  private static let colorTables: (red: [UInt8], green: [UInt8], blue: [UInt8]) = {
    func channel(_ center: Float) -> [UInt8] {
      let ramp = (1..<256).map { 1.5 - abs(4 * Float($0) / 255 - center) }
      return [0] + ramp.map { UInt8((min(max($0, 0), 1) * 255).rounded()) }
    }
    return (channel(3), channel(2), channel(1))
  }()

  /// Stretches the image to the model input without padding, matching Ultralytics depth validation and calibration.
  override var imageCropAndScaleOption: VNImageCropAndScaleOption { .scaleFill }

  override func processObservations(for request: VNRequest, _ error: Error?) {
    markInferenceEnd()
    let depthMap = featureArrays(request).first.flatMap { postProcessDepth($0) }
    updateTime()
    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], depthMap: depthMap, speed: t2,
      fps: 1 / t4, names: labels)
    applyTimingBreakdown(&result, smoothed: true)
    result.originalImage = currentOriginalImage
    currentOnResultsListener?.on(result: result)
  }

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    var depthMap: DepthMap?
    if perform(
      request, with: makeRequestHandler(for: image), errorMessage: "Depth estimation failed")
    {
      markInferenceEnd()
      depthMap = featureArrays(request).first.flatMap { postProcessDepth($0) }
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], depthMap: depthMap, speed: 0, names: labels)
    result.speed = finishTiming(notify: false)
    applyTimingBreakdown(&result)
    result.annotatedImage = drawYOLOSemanticSegmentation(
      ciImage: image, semanticMask: depthMap?.image, opacity: 0.6)
    if capturesOriginalImage {
      result.originalImage = UIImage(ciImage: image)
    }
    return result
  }

  func postProcessDepth(_ output: MLMultiArray) -> DepthMap? {
    let shape = output.shape.map(\.intValue)
    let strides = output.strides.map(\.intValue)
    guard shape.count >= 2, shape.dropLast(2).allSatisfy({ $0 == 1 }) else {
      YOLOLog.error(
        "Invalid depth output shape: expected [1, 1, H, W] or [H, W], got \(output.shape)")
      return nil
    }

    let height = shape[shape.count - 2]
    let width = shape[shape.count - 1]
    let rowStride = strides[strides.count - 2]
    let colStride = strides[strides.count - 1]
    guard width > 0, height > 0 else { return nil }

    var values = [Float](repeating: 0, count: width * height)
    values.withUnsafeMutableBufferPointer { destination in
      if output.dataType == .float32, colStride == 1 {
        let source = output.dataPointer.assumingMemoryBound(to: Float.self)
        for y in 0..<height {
          destination.baseAddress!.advanced(by: y * width).update(
            from: source.advanced(by: y * rowStride), count: width)
        }
      } else {
        for y in 0..<height {
          let sourceRow = y * rowStride
          let destinationRow = y * width
          for x in 0..<width {
            destination[destinationRow + x] = output[sourceRow + x * colStride].floatValue
          }
        }
      }
    }

    let positiveValues = vDSP.threshold(
      values, to: .leastNonzeroMagnitude, with: .zeroFill)
    var zero: Float = 0
    var lowCount: vDSP_Length = 0
    var highCount: vDSP_Length = 0
    var scratch = [Float](repeating: 0, count: values.count)
    vDSP_vclipc(
      positiveValues, 1, &zero, &zero, &scratch, 1, vDSP_Length(values.count), &lowCount,
      &highCount)
    let validCount = Int(lowCount + highCount)
    guard validCount > 0 else { return nil }
    vDSP_vcmprs(values, 1, positiveValues, 1, &scratch, 1, vDSP_Length(values.count))
    var minDepth: Float = 0
    var maxDepth: Float = 0
    vDSP_minv(scratch, 1, &minDepth, vDSP_Length(validCount))
    vDSP_maxv(scratch, 1, &maxDepth, vDSP_Length(validCount))
    guard minDepth.isFinite, maxDepth.isFinite else { return nil }

    // Disparity (1/d) 2nd/98th percentiles from ~4k evenly strided valid values. Ultralytics Python samples ~64k, but
    // sorting that many costs ~3 ms per frame here, and 4k moves the percentiles by under 0.2%.
    let step = Swift.max(1, validCount / 4096)
    var disparities = [Float](repeating: 0, count: (validCount + step - 1) / step)
    var one: Float = 1
    vDSP_svdiv(&one, scratch, step, &disparities, 1, vDSP_Length(disparities.count))
    vDSP.sort(&disparities, sortOrder: .ascending)
    let last = Float(disparities.count - 1)
    return DepthMap(
      values: values,
      width: width,
      height: height,
      minDepth: minDepth,
      maxDepth: maxDepth,
      image: colorizeDepth(
        values, width: width, height: height, low: disparities[Int((0.02 * last).rounded())],
        high: disparities[Int((0.98 * last).rounded())])
    )
  }

  /// Maps disparity linearly from `low` to `high` onto the jet table (near = warm), invalid (d <= 0) pixels black.
  private func colorizeDepth(
    _ values: [Float], width: Int, height: Int, low: Float, high: Float
  ) -> CGImage? {
    let count = vDSP_Length(values.count)
    let valid = vDSP.add(
      0.5, vDSP.threshold(values, to: .leastNonzeroMagnitude, with: .signedConstant(0.5)))
    var scaled = [Float](repeating: 0, count: values.count)
    var pixels = Data(count: values.count * 4)
    scaled.withUnsafeMutableBufferPointer { buffer in
      // Table index = clip(255 (1/d - low) / (high - low), 1, 255), zeroed where d <= 0
      let index = buffer.baseAddress!
      var one: Float = 1
      var scale = 255 / Swift.max(high - low, 1e-6)
      var offset = -low * scale
      var lower: Float = 1
      var upper: Float = 255
      vDSP_svdiv(&one, values, 1, index, 1, count)
      vDSP_vsmsa(index, 1, &scale, &offset, index, 1, count)
      vDSP_vclip(index, 1, &lower, &upper, index, 1, count)
      vDSP_vmul(index, 1, valid, 1, index, 1, count)

      let planeData = UnsafeMutablePointer<UInt8>.allocate(capacity: values.count * 4)
      defer { planeData.deallocate() }
      vDSP_vfixu8(index, 1, planeData, 1, count)
      var indices = vImage_Buffer(
        data: planeData, height: vImagePixelCount(height), width: vImagePixelCount(width),
        rowBytes: width)

      var red = vImage_Buffer(
        data: planeData + values.count, height: vImagePixelCount(height),
        width: vImagePixelCount(width), rowBytes: width)
      var green = vImage_Buffer(
        data: planeData + values.count * 2, height: vImagePixelCount(height),
        width: vImagePixelCount(width), rowBytes: width)
      var blue = vImage_Buffer(
        data: planeData + values.count * 3, height: vImagePixelCount(height),
        width: vImagePixelCount(width), rowBytes: width)
      var redTable = Self.colorTables.red
      var greenTable = Self.colorTables.green
      var blueTable = Self.colorTables.blue
      vImageTableLookUp_Planar8(&indices, &red, &redTable, vImage_Flags(kvImageNoFlags))
      vImageTableLookUp_Planar8(&indices, &green, &greenTable, vImage_Flags(kvImageNoFlags))
      vImageTableLookUp_Planar8(&indices, &blue, &blueTable, vImage_Flags(kvImageNoFlags))
      memset(planeData, 0xFF, values.count)  // the index plane becomes the opaque alpha plane

      pixels.withUnsafeMutableBytes { rawBuffer in
        var rgba = vImage_Buffer(
          data: rawBuffer.baseAddress, height: vImagePixelCount(height),
          width: vImagePixelCount(width), rowBytes: width * 4)
        vImageConvert_Planar8toARGB8888(
          &red, &green, &blue, &indices, &rgba, vImage_Flags(kvImageNoFlags))
      }
    }
    return makeRGBAImage(from: pixels, width: width, height: height)
  }
}
