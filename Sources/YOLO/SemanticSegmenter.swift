// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, implementing semantic segmentation functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://www.ultralytics.com/license
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
    let semanticMask = firstFeatureArray(request).flatMap { postProcessSemantic($0) }
    self.updateTime()
    let result = YOLOResult(
      orig_shape: inputSize, boxes: [], semanticMask: semanticMask, speed: self.t2,
      fps: 1 / self.t4, names: labels)
    self.currentOnResultsListener?.on(result: result)
  }

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    self.inputSize = CGSize(width: image.extent.width, height: image.extent.height)
    let start = Date()
    var semanticMask: SemanticMask?

    do {
      try requestHandler.perform([request])
      semanticMask = firstFeatureArray(request).flatMap { postProcessSemantic($0) }
    } catch {
      YOLOLog.error("Semantic segmentation failed: \(error)")
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], semanticMask: semanticMask,
      speed: Date().timeIntervalSince(start), names: labels)
    result.annotatedImage = drawYOLOSemanticSegmentation(
      ciImage: image, semanticMask: semanticMask?.maskImage)
    return result
  }

  private func firstFeatureArray(_ request: VNRequest) -> MLMultiArray? {
    (request.results as? [VNCoreMLFeatureValueObservation])?.first?.featureValue.multiArrayValue
  }

  func postProcessSemantic(_ logits: MLMultiArray) -> SemanticMask? {
    let shape = logits.shape.map { $0.intValue }
    let strides = logits.strides.map { $0.intValue }
    guard shape.count == 4, shape[0] == 1 else {
      YOLOLog.error(
        "Invalid semantic output shape: expected logits [1, C, H, W] or [1, H, W, C], got \(logits.shape)"
      )
      return nil
    }

    let isNCHW = shape[1] <= shape[3] || shape[1] == labels.count
    let classCount = isNCHW ? shape[1] : shape[3]
    let maskHeight = isNCHW ? shape[2] : shape[1]
    let maskWidth = isNCHW ? shape[3] : shape[2]
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

    let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
    let outCount = outputWidth * outputHeight
    var classMap = [Int](repeating: 0, count: outCount)
    var pixels = Data(count: outCount * 4)
    let colors = semanticColors(classCount: classCount == 1 ? 2 : classCount)
    let binaryThreshold = classCount == 1 ? singleChannelThreshold(pointer, count: logits.count) : 0
    let classStride = isNCHW ? strides[1] : strides[3]
    let rowStride = isNCHW ? strides[2] : strides[1]
    let colStride = isNCHW ? strides[3] : strides[2]

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

    // Phase 2 — paint the RGBA buffer from the resolved class map in one sequential sweep.
    pixels.withUnsafeMutableBytes { rawBuffer in
      let pixelBuffer = rawBuffer.bindMemory(to: UInt8.self)
      for i in 0..<outCount {
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
