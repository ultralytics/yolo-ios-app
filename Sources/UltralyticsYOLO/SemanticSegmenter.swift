// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, implementing semantic segmentation functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  SemanticSegmenter extends BasePredictor to produce a dense class map for the full scene without separating
//  individual object instances. It supports both [1, C, H, W] and [1, H, W, C] tensor layouts, removes letterbox
//  padding via the input mask crop rect, and renders a color overlay for visualization.

import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO semantic segmentation models that produce dense class maps.
public final class SemanticSegmenter: BasePredictor, @unchecked Sendable {
  private var colorCache: (classCount: Int, colors: [(red: UInt8, green: UInt8, blue: UInt8)])?

  override func processObservations(for request: VNRequest, _ error: Error?) {
    markInferenceEnd()
    let semanticMask = firstFeatureArray(request).flatMap { postProcessSemantic($0) }
    self.updateTime()
    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], semanticMask: semanticMask, speed: self.t2,
      fps: 1 / self.t4, names: labels)
    applyTimingBreakdown(&result)
    result.originalImage = currentOriginalImage
    self.currentOnResultsListener?.on(result: result)
  }

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    var semanticMask: SemanticMask?
    let requestHandler = makeRequestHandler(for: image)
    if perform(request, with: requestHandler, errorMessage: "Semantic segmentation failed") {
      markInferenceEnd()
      semanticMask = firstFeatureArray(request).flatMap { postProcessSemantic($0) }
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], semanticMask: semanticMask,
      speed: 0, names: labels)
    result.speed = finishTiming(notify: false)  // before drawing: annotation is excluded from timings
    applyTimingBreakdown(&result)
    result.annotatedImage = drawYOLOSemanticSegmentation(
      ciImage: image, semanticMask: semanticMask?.maskImage)
    if capturesOriginalImage {
      result.originalImage = UIImage(ciImage: image)
    }
    return result
  }

  private func firstFeatureArray(_ request: VNRequest) -> MLMultiArray? {
    (request.results as? [VNCoreMLFeatureValueObservation])?.first?.featureValue.multiArrayValue
  }

  func postProcessSemantic(_ logits: MLMultiArray) -> SemanticMask? {
    let shape = logits.shape.map { $0.intValue }
    let strides = logits.strides.map { $0.intValue }
    // In-graph-ArgMax exports emit a [1, H, W] class map directly; legacy exports emit 4D float logits
    let isClassMap = shape.count == 3 && shape[0] == 1
    guard isClassMap || (shape.count == 4 && shape[0] == 1) else {
      YOLOLog.error(
        "Invalid semantic output shape: expected class map [1, H, W] or logits [1, C, H, W] / [1, H, W, C], got \(logits.shape)"
      )
      return nil
    }

    let isNCHW = isClassMap || shape[1] <= shape[3] || shape[1] == labels.count
    let classCount = isClassMap ? max(labels.count, 2) : (isNCHW ? shape[1] : shape[3])
    let maskHeight = isClassMap ? shape[1] : (isNCHW ? shape[2] : shape[1])
    let maskWidth = isClassMap ? shape[2] : (isNCHW ? shape[3] : shape[2])
    guard classCount > 0, maskWidth > 0, maskHeight > 0 else { return nil }
    if labels.isEmpty {
      YOLOLog.warning("Semantic output axis inferred without labels from shape: \(logits.shape)")
    }

    let crop = inputMaskCropRect(
      maskWidth: maskWidth, maskHeight: maskHeight, inputSize: inputSize,
      modelInputSize: modelInputSize)
    let bounds = CGRect(x: 0, y: 0, width: maskWidth, height: maskHeight)
    let outputRect = (crop ?? bounds).intersection(bounds).integral
    let outputX = Int(outputRect.minX)
    let outputY = Int(outputRect.minY)
    let outputWidth = Int(outputRect.width)
    let outputHeight = Int(outputRect.height)
    guard outputWidth > 0, outputHeight > 0 else { return nil }

    let outCount = outputWidth * outputHeight
    var classMap = [Int](repeating: 0, count: outCount)
    var pixels = Data(count: outCount * 4)
    let colors = semanticColors(classCount: classCount == 1 ? 2 : classCount)
    let classStride = isClassMap ? 0 : (isNCHW ? strides[1] : strides[3])
    let rowStride = isClassMap ? strides[1] : (isNCHW ? strides[2] : strides[1])
    let colStride = isClassMap ? strides[2] : (isNCHW ? strides[3] : strides[2])

    if isClassMap {
      // The NPU already did the argmax - just read the per-pixel class indices (dtype depends on the runtime)
      let readIndex: (Int) -> Int
      switch logits.dataType {
      case .int32:
        let p = logits.dataPointer.assumingMemoryBound(to: Int32.self)
        readIndex = { Int(p[$0]) }
      default:
        let p = logits.dataPointer.assumingMemoryBound(to: Float.self)
        readIndex = { Int(p[$0]) }
      }
      for y in 0..<outputHeight {
        let srcRow = (y + outputY) * rowStride + outputX * colStride
        let outRow = y * outputWidth
        for x in 0..<outputWidth {
          classMap[outRow + x] = min(max(readIndex(srcRow + x * colStride), 0), classCount - 1)
        }
      }
      return finishSemanticMask(
        classMap: classMap, pixels: &pixels, colors: colors,
        outputWidth: outputWidth, outputHeight: outputHeight)
    }

    let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
    let binaryThreshold = classCount == 1 ? singleChannelThreshold(pointer, count: logits.count) : 0

    // Phase 1 — argmax over classes into `classMap`, iterating class-major: for each class plane sweep the output
    // pixels, keeping a running best score/index. For NCHW (the YOLO export layout) each plane is contiguous, so
    // reads stay sequential — far cheaper than a per-pixel inner loop over classes, which reads H*W apart. Ties
    // keep the lowest class index (argmax semantics). Strides keep it correct for an NHWC layout too.
    if classCount == 1 {
      for y in 0..<outputHeight {
        let srcRow = (y + outputY) * rowStride + outputX * colStride
        let outRow = y * outputWidth
        for x in 0..<outputWidth {
          classMap[outRow + x] = pointer[srcRow + x * colStride] > binaryThreshold ? 1 : 0
        }
      }
    } else {
      var best = [Float](repeating: -.greatestFiniteMagnitude, count: outCount)
      classMap.withUnsafeMutableBufferPointer { cm in
        best.withUnsafeMutableBufferPointer { bb in
          for c in 0..<classCount {
            let classBase = c * classStride
            for y in 0..<outputHeight {
              let srcRow = classBase + (y + outputY) * rowStride + outputX * colStride
              let outRow = y * outputWidth
              for x in 0..<outputWidth {
                let score = pointer[srcRow + x * colStride]
                let oi = outRow + x
                if score > bb[oi] {
                  bb[oi] = score
                  cm[oi] = c
                }
              }
            }
          }
        }
      }
    }

    return finishSemanticMask(
      classMap: classMap, pixels: &pixels, colors: colors,
      outputWidth: outputWidth, outputHeight: outputHeight)
  }

  /// Paints the RGBA buffer from a resolved class map in one sequential sweep and packages the result.
  private func finishSemanticMask(
    classMap: [Int], pixels: inout Data, colors: [(red: UInt8, green: UInt8, blue: UInt8)],
    outputWidth: Int, outputHeight: Int
  ) -> SemanticMask {
    pixels.withUnsafeMutableBytes { rawBuffer in
      let pixelBuffer = rawBuffer.bindMemory(to: UInt8.self)
      for i in 0..<(outputWidth * outputHeight) {
        writeColor(colors[classMap[i]], into: pixelBuffer, at: i * 4)
      }
    }

    return SemanticMask(
      classMap: classMap,
      width: outputWidth,
      height: outputHeight,
      maskImage: makeImage(fromRGBA: pixels, width: outputWidth, height: outputHeight))
  }

  private func singleChannelThreshold(_ pointer: UnsafeMutablePointer<Float>, count: Int) -> Float {
    var minValue = Float.greatestFiniteMagnitude
    var maxValue = -Float.greatestFiniteMagnitude
    for index in 0..<count {
      let value = pointer[index]
      minValue = min(minValue, value)
      maxValue = max(maxValue, value)
    }
    return minValue >= 0 && maxValue <= 1 ? 0.5 : 0
  }

  private func semanticColors(classCount: Int) -> [(red: UInt8, green: UInt8, blue: UInt8)] {
    if let colorCache, colorCache.classCount == classCount {
      return colorCache.colors
    }

    let colors = (0..<classCount).map { classIndex in
      let color = ultralyticsColors[classIndex % ultralyticsColors.count]
      var red: CGFloat = 0
      var green: CGFloat = 0
      var blue: CGFloat = 0
      color.getRed(&red, green: &green, blue: &blue, alpha: nil)
      return (UInt8(red * 255), UInt8(green * 255), UInt8(blue * 255))
    }
    colorCache = (classCount, colors)
    return colors
  }

  private func writeColor(
    _ color: (red: UInt8, green: UInt8, blue: UInt8),
    into pixels: UnsafeMutableBufferPointer<UInt8>,
    at offset: Int
  ) {
    pixels[offset] = color.red
    pixels[offset + 1] = color.green
    pixels[offset + 2] = color.blue
    pixels[offset + 3] = 255
  }

  private func makeImage(fromRGBA pixels: Data, width: Int, height: Int) -> CGImage? {
    guard let provider = CGDataProvider(data: pixels as CFData) else { return nil }
    return CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
      provider: provider,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent)
  }
}
