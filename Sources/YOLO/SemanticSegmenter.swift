// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO semantic segmentation models that produce dense class maps.
public final class SemanticSegmenter: BasePredictor, @unchecked Sendable {

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
      YOLOLog.error("Invalid semantic output shape: \(logits.shape)")
      return nil
    }

    let isNCHW = shape[1] <= shape[3] || shape[1] == labels.count
    let classCount = isNCHW ? shape[1] : shape[3]
    let maskHeight = isNCHW ? shape[2] : shape[1]
    let maskWidth = isNCHW ? shape[3] : shape[2]
    guard classCount > 0, maskWidth > 0, maskHeight > 0 else { return nil }

    let crop = inputMaskCropRect(
      maskWidth: maskWidth, maskHeight: maskHeight, inputSize: inputSize,
      modelInputSize: modelInputSize)
    let bounds = CGRect(x: 0, y: 0, width: maskWidth, height: maskHeight)
    let sampleRect = (crop ?? bounds).intersection(bounds)
    let outputWidth = max(
      Int((sampleRect.width / CGFloat(maskWidth) * CGFloat(modelInputSize.width)).rounded()), 1)
    let outputHeight = max(
      Int((sampleRect.height / CGFloat(maskHeight) * CGFloat(modelInputSize.height)).rounded()), 1)
    guard outputWidth > 0, outputHeight > 0 else { return nil }

    let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
    var classMap = [Int](repeating: 0, count: outputWidth * outputHeight)
    var pixels = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)
    let colors = semanticColors(classCount: classCount)
    let scaleX = Float(sampleRect.width) / Float(outputWidth)
    let scaleY = Float(sampleRect.height) / Float(outputHeight)
    let originX = Float(sampleRect.minX)
    let originY = Float(sampleRect.minY)

    for y in 0..<outputHeight {
      let sourceY = originY + (Float(y) + 0.5) * scaleY - 0.5
      for x in 0..<outputWidth {
        let sourceX = originX + (Float(x) + 0.5) * scaleX - 0.5
        let classIndex = bestClass(
          pointer: pointer, shape: shape, strides: strides, classCount: classCount,
          x: sourceX, y: sourceY, isNCHW: isNCHW)
        let outIndex = y * outputWidth + x
        classMap[outIndex] = classIndex
        writeColor(colors[classIndex], into: &pixels, at: outIndex * 4)
      }
    }

    return SemanticMask(
      classMap: classMap,
      width: outputWidth,
      height: outputHeight,
      maskImage: makeImage(fromRGBA: pixels, width: outputWidth, height: outputHeight))
  }

  private func bestClass(
    pointer: UnsafeMutablePointer<Float>,
    shape: [Int],
    strides: [Int],
    classCount: Int,
    x: Float,
    y: Float,
    isNCHW: Bool
  ) -> Int {
    if classCount == 1 {
      return 0
    }

    var bestIndex = 0
    var bestScore = -Float.greatestFiniteMagnitude
    for c in 0..<classCount {
      let score = bilinearValue(
        pointer: pointer, shape: shape, strides: strides, classIndex: c, x: x, y: y, isNCHW: isNCHW)
      if score > bestScore {
        bestScore = score
        bestIndex = c
      }
    }
    return bestIndex
  }

  private func bilinearValue(
    pointer: UnsafeMutablePointer<Float>,
    shape: [Int],
    strides: [Int],
    classIndex: Int,
    x: Float,
    y: Float,
    isNCHW: Bool
  ) -> Float {
    let width = isNCHW ? shape[3] : shape[2]
    let height = isNCHW ? shape[2] : shape[1]
    let x0 = min(max(Int(floor(x)), 0), width - 1)
    let y0 = min(max(Int(floor(y)), 0), height - 1)
    let x1 = min(x0 + 1, width - 1)
    let y1 = min(y0 + 1, height - 1)
    let wx = min(max(x - Float(x0), 0), 1)
    let wy = min(max(y - Float(y0), 0), 1)
    let top =
      value(
        pointer: pointer, strides: strides, classIndex: classIndex, x: x0, y: y0, isNCHW: isNCHW)
      * (1 - wx)
      + value(
        pointer: pointer, strides: strides, classIndex: classIndex, x: x1, y: y0, isNCHW: isNCHW)
      * wx
    let bottom =
      value(
        pointer: pointer, strides: strides, classIndex: classIndex, x: x0, y: y1, isNCHW: isNCHW)
      * (1 - wx)
      + value(
        pointer: pointer, strides: strides, classIndex: classIndex, x: x1, y: y1, isNCHW: isNCHW)
      * wx
    return top * (1 - wy) + bottom * wy
  }

  private func value(
    pointer: UnsafeMutablePointer<Float>,
    strides: [Int],
    classIndex: Int,
    x: Int,
    y: Int,
    isNCHW: Bool
  ) -> Float {
    isNCHW
      ? pointer[classIndex * strides[1] + y * strides[2] + x * strides[3]]
      : pointer[y * strides[1] + x * strides[2] + classIndex * strides[3]]
  }

  private func semanticColors(classCount: Int) -> [(red: UInt8, green: UInt8, blue: UInt8)] {
    (0..<classCount).map { classIndex in
      let color = ultralyticsColors[classIndex % ultralyticsColors.count]
      var red: CGFloat = 0
      var green: CGFloat = 0
      var blue: CGFloat = 0
      color.getRed(&red, green: &green, blue: &blue, alpha: nil)
      return (UInt8(red * 255), UInt8(green * 255), UInt8(blue * 255))
    }
  }

  private func writeColor(
    _ color: (red: UInt8, green: UInt8, blue: UInt8), into pixels: inout [UInt8], at offset: Int
  ) {
    pixels[offset] = color.red
    pixels[offset + 1] = color.green
    pixels[offset + 2] = color.blue
    pixels[offset + 3] = 255
  }

  private func makeImage(fromRGBA pixels: [UInt8], width: Int, height: Int) -> CGImage? {
    let data = Data(pixels)
    guard let provider = CGDataProvider(data: data as CFData) else { return nil }
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
