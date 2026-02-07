// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Accelerate
import CoreML
import Foundation
import Vision

/// Color palette for segmentation masks.
private let segmentationColors: [(red: UInt8, green: UInt8, blue: UInt8)] = [
  (4, 42, 255), (11, 219, 235), (243, 243, 243), (0, 223, 183),
  (17, 31, 104), (255, 111, 221), (255, 68, 79), (204, 237, 0),
  (0, 243, 68), (189, 0, 255), (0, 180, 255), (221, 0, 186),
  (0, 255, 255), (38, 192, 0), (1, 255, 179), (125, 36, 255),
  (123, 0, 104), (255, 27, 108), (252, 109, 47), (162, 255, 11),
]

/// Wrapper to pass float pointer across concurrency boundaries.
private final class FloatPointerWrapper: @unchecked Sendable {
  let pointer: UnsafeMutablePointer<Float>
  init(_ pointer: UnsafeMutablePointer<Float>) { self.pointer = pointer }
}

/// Specialized predictor for YOLO segmentation models.
public class Segmenter: BasePredictor {

  override func processResults() -> YOLOResult {
    guard let request = visionRequest,
      let results = request.results as? [VNCoreMLFeatureValueObservation],
      results.count == 2
    else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
    }

    guard let out0 = results[0].featureValue.multiArrayValue,
      let out1 = results[1].featureValue.multiArrayValue
    else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
    }

    let pred: MLMultiArray
    let masks: MLMultiArray
    if checkShapeDimensions(of: out0) == 4 {
      masks = out0
      pred = out1
    } else {
      masks = out1
      pred = out0
    }

    let detectedObjects = postProcessSegment(
      feature: pred,
      confidenceThreshold: Float(configuration.confidenceThreshold),
      iouThreshold: Float(configuration.iouThreshold))

    var boxes = [Box]()
    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)
    let inputWidth = Int(inputSize.width)
    let inputHeight = Int(inputSize.height)

    let limitedObjects = detectedObjects.prefix(configuration.maxDetections)
    for p in limitedObjects {
      let box = p.0
      let rect = CGRect(
        x: box.minX / modelWidth, y: box.minY / modelHeight,
        width: box.width / modelWidth, height: box.height / modelHeight)
      let label = self.labels[p.1]
      let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)
      boxes.append(Box(index: p.1, cls: label, conf: p.2, xywh: xywh, xywhn: rect))
    }

    let processedMasks = generateCombinedMaskImage(
      detectedObjects: detectedObjects,
      protos: masks,
      inputWidth: modelInputSize.width,
      inputHeight: modelInputSize.height,
      threshold: 0.5
    )

    let maskResults: Masks? =
      processedMasks.map { Masks(masks: $0.1, combinedMask: $0.0) }

    return YOLOResult(
      orig_shape: inputSize, boxes: boxes, masks: maskResults,
      speed: t1, fps: t4 > 0 ? 1 / t4 : nil, names: labels)
  }

  private func postProcessSegment(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let numAnchors = feature.shape[2].intValue
    let numFeatures = feature.shape[1].intValue
    let maskConfidenceLength = 32
    let numClasses = numFeatures - 4 - maskConfidenceLength

    final class ResultsWrapper: @unchecked Sendable {
      private let lock = NSLock()
      private var results: [(CGRect, Int, Float, MLMultiArray)]

      init(capacity: Int) {
        results = []
        results.reserveCapacity(capacity)
      }

      func append(_ result: (CGRect, Int, Float, MLMultiArray)) {
        lock.lock()
        results.append(result)
        lock.unlock()
      }

      func getResults() -> [(CGRect, Int, Float, MLMultiArray)] { results }
    }

    let estimatedCapacity = min(numAnchors / 10, 100)
    let resultsWrapper = ResultsWrapper(capacity: estimatedCapacity)
    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let pointerWrapper = FloatPointerWrapper(featurePointer)

    DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
      let x = pointerWrapper.pointer[j]
      let y = pointerWrapper.pointer[numAnchors + j]
      let width = pointerWrapper.pointer[2 * numAnchors + j]
      let height = pointerWrapper.pointer[3 * numAnchors + j]
      let boxX = CGFloat(x - width / 2)
      let boxY = CGFloat(y - height / 2)
      let boundingBox = CGRect(x: boxX, y: boxY, width: CGFloat(width), height: CGFloat(height))

      let localClassProbs = UnsafeMutableBufferPointer<Float>.allocate(capacity: numClasses)
      defer { localClassProbs.deallocate() }

      vDSP_mtrans(
        pointerWrapper.pointer + 4 * numAnchors + j, numAnchors,
        localClassProbs.baseAddress!, 1, 1, vDSP_Length(numClasses))

      var maxClassValue: Float = 0
      var maxClassIndex: vDSP_Length = 0
      vDSP_maxvi(
        localClassProbs.baseAddress!, 1, &maxClassValue, &maxClassIndex, vDSP_Length(numClasses))

      if maxClassValue > confidenceThreshold {
        guard
          let maskProbs = try? MLMultiArray(
            shape: [NSNumber(value: maskConfidenceLength)], dataType: .float32)
        else { return }

        let maskProbsPointer = pointerWrapper.pointer + (4 + numClasses) * numAnchors + j
        let maskProbsData = maskProbs.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<maskConfidenceLength {
          maskProbsData[i] = maskProbsPointer[i * numAnchors]
        }
        resultsWrapper.append((boundingBox, Int(maxClassIndex), maxClassValue, maskProbs))
      }
    }

    let collectedResults = resultsWrapper.getResults()

    // NMS by class
    var classBuckets: [Int: [(CGRect, Int, Float, MLMultiArray)]] = [:]
    for result in collectedResults {
      classBuckets[result.1, default: []].append(result)
    }

    var selectedBoxesAndFeatures: [(CGRect, Int, Float, MLMultiArray)] = []
    selectedBoxesAndFeatures.reserveCapacity(collectedResults.count)

    for (_, classResults) in classBuckets {
      let boxesOnly = classResults.map { $0.0 }
      let scoresOnly = classResults.map { $0.2 }
      let selectedIndices = nonMaxSuppression(
        boxes: boxesOnly, scores: scoresOnly, threshold: iouThreshold)
      for idx in selectedIndices {
        selectedBoxesAndFeatures.append(classResults[idx])
      }
    }
    return selectedBoxesAndFeatures
  }

  private func checkShapeDimensions(of multiArray: MLMultiArray) -> Int {
    multiArray.shape.count
  }

  private func generateCombinedMaskImage(
    detectedObjects: [(CGRect, Int, Float, MLMultiArray)],
    protos: MLMultiArray,
    inputWidth: Int,
    inputHeight: Int,
    threshold: Float
  ) -> (CGImage?, [[[Float]]])? {
    let maskHeight = protos.shape[2].intValue
    let maskWidth = protos.shape[3].intValue
    let maskChannels = protos.shape[1].intValue
    guard protos.shape.count == 4, protos.shape[0].intValue == 1,
      maskHeight > 0, maskWidth > 0, maskChannels > 0
    else { return nil }

    let protosPointer = protos.dataPointer.assumingMemoryBound(to: Float.self)
    let HW = maskHeight * maskWidth
    let N = detectedObjects.count

    var coeffsArray = [Float](repeating: 0, count: N * maskChannels)
    for i in 0..<N {
      let coeffsMLArray = detectedObjects[i].3
      let coeffsPtr = coeffsMLArray.dataPointer.assumingMemoryBound(to: Float.self)
      for c in 0..<maskChannels {
        coeffsArray[i * maskChannels + c] = coeffsPtr[c]
      }
    }

    var combinedMask = [Float](repeating: 0, count: N * HW)
    coeffsArray.withUnsafeBufferPointer { Abuf in
      combinedMask.withUnsafeMutableBufferPointer { Cbuf in
        vDSP_mmul(
          Abuf.baseAddress!, 1, protosPointer, 1, Cbuf.baseAddress!, 1,
          vDSP_Length(N), vDSP_Length(HW), vDSP_Length(maskChannels))
      }
    }

    let indexedObjects: [(Int, CGRect, Int, Float)] =
      detectedObjects.enumerated().map { (i, obj) in (i, obj.0, obj.1, obj.2) }
    let sortedObjects = indexedObjects.sorted { $0.3 < $1.3 }

    var mergedPixels = [UInt8](repeating: 0, count: HW * 4)
    let scaleX = Float(maskWidth) / Float(inputWidth)
    let scaleY = Float(maskHeight) / Float(inputHeight)

    var probabilityMasks = Array(
      repeating: Array(
        repeating: Array(repeating: Float(0.0), count: maskWidth), count: maskHeight),
      count: N)

    for (originalIndex, box, classID, _) in sortedObjects {
      let boxX1 = max(0, min(Int(Float(box.minX) * scaleX), maskWidth - 1))
      let boxX2 = max(0, min(Int(Float(box.maxX) * scaleX), maskWidth - 1))
      let boxY1 = max(0, min(Int(Float(box.minY) * scaleY), maskHeight - 1))
      let boxY2 = max(0, min(Int(Float(box.maxY) * scaleY), maskHeight - 1))
      let startIdx = originalIndex * HW
      let colorIndex = classID % segmentationColors.count
      let color = segmentationColors[colorIndex]

      for y in boxY1...boxY2 {
        for x in boxX1...boxX2 {
          let px = y * maskWidth + x
          let maskVal = combinedMask[startIdx + px]
          if maskVal > threshold {
            let pixIndex = px * 4
            mergedPixels[pixIndex] = color.red
            mergedPixels[pixIndex + 1] = color.green
            mergedPixels[pixIndex + 2] = color.blue
            mergedPixels[pixIndex + 3] = 255
          }
        }
      }
    }

    for i in 0..<N {
      let startIdx = i * HW
      for k in 0..<HW {
        probabilityMasks[i][k / maskWidth][k % maskWidth] = combinedMask[startIdx + k]
      }
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    guard
      let providerRef = CGDataProvider(
        data: NSData(bytes: &mergedPixels, length: mergedPixels.count))
    else { return nil }
    guard
      let mergedCGImage = CGImage(
        width: maskWidth, height: maskHeight,
        bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: maskWidth * 4,
        space: colorSpace, bitmapInfo: bitmapInfo, provider: providerRef,
        decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    else { return nil }

    return (mergedCGImage, probabilityMasks)
  }
}
