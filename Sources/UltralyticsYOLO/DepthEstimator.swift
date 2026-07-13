// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Accelerate
import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for monocular depth models that produce one metric distance per pixel.
public final class DepthEstimator: BasePredictor, @unchecked Sendable {
  private static let colorTables: (red: [UInt8], green: [UInt8], blue: [UInt8]) = {
    var red = [UInt8](repeating: 0, count: 256)
    var green = red
    var blue = red
    let stops: [(Float, Float, Float)] = [
      (48, 18, 59), (50, 100, 200), (40, 190, 140), (245, 210, 60), (180, 20, 40),
    ]
    for index in 0..<256 {
      let position = Float(index) / 255 * Float(stops.count - 1)
      let lower = min(Int(position), stops.count - 2)
      let fraction = position - Float(lower)
      let a = stops[lower]
      let b = stops[lower + 1]
      red[index] = UInt8(a.0 + (b.0 - a.0) * fraction)
      green[index] = UInt8(a.1 + (b.1 - a.1) * fraction)
      blue[index] = UInt8(a.2 + (b.2 - a.2) * fraction)
    }
    return (red, green, blue)
  }()

  override func processObservations(for request: VNRequest, _ error: Error?) {
    markInferenceEnd()
    let depthMap = firstFeatureArray(request).flatMap { postProcessDepth($0) }
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
      depthMap = firstFeatureArray(request).flatMap { postProcessDepth($0) }
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], depthMap: depthMap, speed: 0, names: labels)
    result.speed = finishTiming(notify: false)
    applyTimingBreakdown(&result)
    result.annotatedImage = drawYOLOSemanticSegmentation(
      ciImage: image, semanticMask: depthMap?.image)
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

    let bounds = CGRect(x: 0, y: 0, width: width, height: height)
    let rect =
      (inputMaskCropRect(
        maskWidth: width, maskHeight: height, inputSize: inputSize,
        modelInputSize: modelInputSize) ?? bounds).intersection(bounds).integral
    let x0 = Int(rect.minX)
    let y0 = Int(rect.minY)
    let outputWidth = Int(rect.width)
    let outputHeight = Int(rect.height)
    guard outputWidth > 0, outputHeight > 0 else { return nil }

    var values = [Float](repeating: 0, count: outputWidth * outputHeight)
    values.withUnsafeMutableBufferPointer { destination in
      if output.dataType == .float32, colStride == 1 {
        let source = output.dataPointer.assumingMemoryBound(to: Float.self)
        for y in 0..<outputHeight {
          destination.baseAddress!.advanced(by: y * outputWidth).update(
            from: source.advanced(by: (y + y0) * rowStride + x0), count: outputWidth)
        }
      } else {
        for y in 0..<outputHeight {
          let sourceRow = (y + y0) * rowStride + x0 * colStride
          let destinationRow = y * outputWidth
          for x in 0..<outputWidth {
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
    return DepthMap(
      values: values,
      width: outputWidth,
      height: outputHeight,
      minDepth: minDepth,
      maxDepth: maxDepth,
      image: colorizeDepth(
        values, width: outputWidth, height: outputHeight, min: minDepth, max: maxDepth)
    )
  }

  private func colorizeDepth(
    _ values: [Float], width: Int, height: Int, min minDepth: Float, max maxDepth: Float
  ) -> CGImage? {
    let low = log(Swift.max(minDepth, 1e-3))
    let range = Swift.max(log(maxDepth) - low, 1e-6)
    var normalized = vDSP.threshold(values, to: minDepth, with: .clampToThreshold)
    var count = Int32(values.count)
    normalized.withUnsafeMutableBufferPointer { destination in
      vvlogf(destination.baseAddress!, destination.baseAddress!, &count)
      var scale = -1 / range
      var offset = log(maxDepth) / range
      vDSP_vsmsa(
        destination.baseAddress!, 1, &scale, &offset, destination.baseAddress!, 1,
        vDSP_Length(values.count))
      var lower: Float = 0
      var upper: Float = 1
      vDSP_vclip(
        destination.baseAddress!, 1, &lower, &upper, destination.baseAddress!, 1,
        vDSP_Length(values.count))
    }

    var pixels = Data(count: values.count * 4)
    normalized.withUnsafeMutableBufferPointer { source in
      let planeData = UnsafeMutablePointer<UInt8>.allocate(capacity: values.count * 4)
      defer { planeData.deallocate() }

      var sourceBuffer = vImage_Buffer(
        data: source.baseAddress, height: vImagePixelCount(height), width: vImagePixelCount(width),
        rowBytes: width * MemoryLayout<Float>.stride)
      var indices = vImage_Buffer(
        data: planeData, height: vImagePixelCount(height), width: vImagePixelCount(width),
        rowBytes: width)
      vImageConvert_PlanarFtoPlanar8(&sourceBuffer, &indices, 1, 0, vImage_Flags(kvImageNoFlags))

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
      memset(planeData, 0xFF, values.count)

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
