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
    let outputRect = (crop ?? bounds).intersection(bounds).integral
    let outputX = Int(outputRect.minX)
    let outputY = Int(outputRect.minY)
    let outputWidth = Int(outputRect.width)
    let outputHeight = Int(outputRect.height)
    guard outputWidth > 0, outputHeight > 0 else { return nil }

    let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
    var classMap = [Int](repeating: 0, count: outputWidth * outputHeight)
    var pixels = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)

    for y in 0..<outputHeight {
      let sourceY = y + outputY
      for x in 0..<outputWidth {
        let sourceX = x + outputX
        let classIndex = bestClass(
          pointer: pointer, shape: shape, strides: strides, classCount: classCount,
          x: sourceX, y: sourceY, isNCHW: isNCHW)
        let outIndex = y * outputWidth + x
        classMap[outIndex] = classIndex
        writeColor(for: classIndex, into: &pixels, at: outIndex * 4)
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
    x: Int,
    y: Int,
    isNCHW: Bool
  ) -> Int {
    if classCount == 1 {
      return 0
    }

    var bestIndex = 0
    var bestScore = -Float.greatestFiniteMagnitude
    for c in 0..<classCount {
      let score =
        isNCHW
        ? pointer[c * strides[1] + y * strides[2] + x * strides[3]]
        : pointer[y * strides[1] + x * strides[2] + c * strides[3]]
      if score > bestScore {
        bestScore = score
        bestIndex = c
      }
    }
    return bestIndex
  }

  private func writeColor(for classIndex: Int, into pixels: inout [UInt8], at offset: Int) {
    let color = ultralyticsColors[classIndex % ultralyticsColors.count]
    var red: CGFloat = 0
    var green: CGFloat = 0
    var blue: CGFloat = 0
    color.getRed(&red, green: &green, blue: &blue, alpha: nil)
    pixels[offset] = UInt8(red * 255)
    pixels[offset + 1] = UInt8(green * 255)
    pixels[offset + 2] = UInt8(blue * 255)
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
