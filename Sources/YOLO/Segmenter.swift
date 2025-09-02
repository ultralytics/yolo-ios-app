// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, implementing instance segmentation functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Segmenter class extends BasePredictor to provide instance segmentation capabilities.
//  Instance segmentation not only detects objects but also identifies the precise pixels
//  belonging to each object. The class processes complex model outputs including prototype masks
//  and detection results, performs non-maximum suppression to filter detections, and combines
//  results into visualizable mask images. It leverages the Accelerate framework for efficient
//  matrix operations and includes parallel processing to optimize performance on mobile devices.
//  The results include both bounding boxes and pixel-level masks that can be overlaid on images.

import Accelerate
@preconcurrency import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO segmentation models that identify objects and their pixel-level masks.
class Segmenter: BasePredictor, @unchecked Sendable {
  var colorsForMask: [(red: UInt8, green: UInt8, blue: UInt8)] = []

  override func processObservations(for request: VNRequest, error: Error?) {
    if let results = request.results as? [VNCoreMLFeatureValueObservation] {
      //            DispatchQueue.main.async { [self] in
      guard results.count == 2 else { return }
      var pred: MLMultiArray
      var masks: MLMultiArray
      guard let out0 = results[0].featureValue.multiArrayValue,
        let out1 = results[1].featureValue.multiArrayValue
      else { return }
      let out0dim = checkShapeDimensions(of: out0)
      _ = checkShapeDimensions(of: out1)
      if out0dim == 4 {
        masks = out0
        pred = out1
      } else {
        masks = out1
        pred = out0
      }
      let detectedObjects = postProcessSegment(
        feature: pred, confidenceThreshold: Float(confidenceThreshold),
        iouThreshold: Float(iouThreshold))

      let detectionsCount = detectedObjects.count
      var boxes: [Box] = []
      boxes.reserveCapacity(detectionsCount)
      var alphas = [CGFloat]()
      alphas.reserveCapacity(detectionsCount)

      let modelWidth = CGFloat(self.modelInputSize.width)
      let modelHeight = CGFloat(self.modelInputSize.height)
      let inputWidth = Int(self.inputSize.width)
      let inputHeight = Int(self.inputSize.height)

      // Pre-calculate alpha constants
      let alphaScale: CGFloat = 0.9 / 0.8  // (1.0 - 0.2)
      let alphaOffset: CGFloat = -0.2 * alphaScale

      let limitedObjects = detectedObjects.prefix(self.numItemsThreshold)
      for p in limitedObjects {
        let box = p.0
        let rect = CGRect(
          x: box.minX / modelWidth, y: box.minY / modelHeight,
          width: box.width / modelWidth, height: box.height / modelHeight)
        let confidence = p.2
        let bestClass = p.1
        let label = self.labels[bestClass]
        let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

        let boxResult = Box(index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
        let alpha = CGFloat(confidence) * alphaScale + alphaOffset
        boxes.append(boxResult)
        alphas.append(alpha)
      }

      DispatchQueue.global(qos: .userInitiated).async {
        guard
          let procceessedMasks = generateCombinedMaskImage(
            detectedObjects: Array(limitedObjects),
            protos: masks,
            inputWidth: self.modelInputSize.width,
            inputHeight: self.modelInputSize.height,
            threshold: 0.5

          ) as? (CGImage?, [[[Float]]])
        else {
          DispatchQueue.main.async { self.isUpdating = false }
          return
        }
        let maskResults = Masks(masks: procceessedMasks.1, combinedMask: procceessedMasks.0)
        let result = YOLOResult(
          orig_shape: self.inputSize, boxes: boxes, masks: maskResults, speed: self.t2,
          fps: 1 / self.t4, names: self.labels)
        self.updateTime()
        self.currentOnResultsListener?.on(result: result)
      }
    }
  }

  private func updateTime() {
    if self.t1 < 10.0 {  // valid dt
      self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
    }
    self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
    self.t3 = CACurrentMediaTime()

    self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)  // t2 seconds to ms

  }

  override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
      return emptyResult
    }

    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    var result = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)

    do {
      try requestHandler.perform([request])
      if let results = request.results as? [VNCoreMLFeatureValueObservation] {
        guard results.count == 2 else {
          return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
        }

        // 1. Parse model outputs
        var pred: MLMultiArray
        var masks: MLMultiArray
        guard let out0 = results[0].featureValue.multiArrayValue,
          let out1 = results[1].featureValue.multiArrayValue
        else {
          return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
        }

        let out0dim = checkShapeDimensions(of: out0)
        _ = checkShapeDimensions(of: out1)
        if out0dim == 4 {
          masks = out0
          pred = out1
        } else {
          masks = out1
          pred = out0
        }

        // 2. Post-process detection results
        let detectedObjects = postProcessSegment(
          feature: pred, confidenceThreshold: Float(self.confidenceThreshold),
          iouThreshold: Float(self.iouThreshold))

        // 3. Construct bounding box information
        let detectionsCount = detectedObjects.count
        var boxes: [Box] = []
        boxes.reserveCapacity(detectionsCount)

        let modelWidth = CGFloat(self.modelInputSize.width)
        let modelHeight = CGFloat(self.modelInputSize.height)
        let inputWidth = Int(inputSize.width)
        let inputHeight = Int(inputSize.height)

        let limitedObjects = detectedObjects.prefix(self.numItemsThreshold)
        for p in limitedObjects {
          let box = p.0
          let rect = CGRect(
            x: box.minX / modelWidth, y: box.minY / modelHeight,
            width: box.width / modelWidth, height: box.height / modelHeight)
          let confidence = p.2
          let bestClass = p.1
          let label = labels[bestClass]
          let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

          let boxResult = Box(
            index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
          boxes.append(boxResult)
        }

        // 4. Generate mask image
        guard
          let processedMasks = generateCombinedMaskImage(
            detectedObjects: Array(limitedObjects),
            protos: masks,
            inputWidth: self.modelInputSize.width,
            inputHeight: self.modelInputSize.height,
            threshold: 0.5
          ) as? (CGImage?, [[[Float]]])
        else {
          return YOLOResult(
            orig_shape: inputSize, boxes: boxes, masks: nil, annotatedImage: nil, speed: 0,
            names: labels)
        }

        // 5. Use the new integrated drawing function to render masks and boxes in a single pass
        let annotatedImage = drawYOLOSegmentationWithBoxes(
          ciImage: image,
          boxes: boxes,
          maskImage: processedMasks.0,
          originalImageSize: inputSize
        )

        // 6. Construct result
        let maskResults: Masks = Masks(masks: processedMasks.1, combinedMask: processedMasks.0)

        // 7. Update timing measurements
        updateTime()

        // 8. Return result
        result = YOLOResult(
          orig_shape: inputSize,
          boxes: boxes,
          masks: maskResults,
          annotatedImage: annotatedImage,
          speed: self.t2,
          fps: 1 / self.t4,
          names: labels
        )

        return result
      }
    } catch {
      print(error)
    }
    return result
  }

  nonisolated func postProcessSegment(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {

    let numAnchors = feature.shape[2].intValue
    let numFeatures = feature.shape[1].intValue
    let boxFeatureLength = 4
    let maskConfidenceLength = 32
    let numClasses = numFeatures - boxFeatureLength - maskConfidenceLength

    // Pre-allocate result arrays with estimated capacity
    let resultsWrapper = ResultsWrapper()
    resultsWrapper.reserveCapacity(min(numAnchors / 10, 100))  // Estimate ~10% detection rate

    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let pointerWrapper = FloatPointerWrapper(featurePointer)
    _ = DispatchQueue(label: "resultsQueue", attributes: .concurrent)
    let resultsLock = NSLock()

    // Pre-allocate reusable arrays outside the loop
    let classProbs = UnsafeMutableBufferPointer<Float>.allocate(capacity: numClasses)
    defer { classProbs.deallocate() }

    DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
      let x = pointerWrapper.pointer[j]
      let y = pointerWrapper.pointer[numAnchors + j]
      let width = pointerWrapper.pointer[2 * numAnchors + j]
      let height = pointerWrapper.pointer[3 * numAnchors + j]

      let boxX = CGFloat(x - width / 2)
      let boxY = CGFloat(y - height / 2)
      let boundingBox = CGRect(x: boxX, y: boxY, width: CGFloat(width), height: CGFloat(height))

      // Use thread-local storage for class probabilities
      let localClassProbs = UnsafeMutableBufferPointer<Float>.allocate(capacity: numClasses)
      defer { localClassProbs.deallocate() }

      vDSP_mtrans(
        pointerWrapper.pointer + 4 * numAnchors + j,
        numAnchors,
        localClassProbs.baseAddress!,
        1,
        1,
        vDSP_Length(numClasses)
      )

      var maxClassValue: Float = 0
      var maxClassIndex: vDSP_Length = 0
      vDSP_maxvi(
        localClassProbs.baseAddress!, 1, &maxClassValue, &maxClassIndex, vDSP_Length(numClasses))

      if maxClassValue > confidenceThreshold {
        // Create MLMultiArray more efficiently
        guard
          let maskProbs = try? MLMultiArray(
            shape: [NSNumber(value: maskConfidenceLength)], dataType: .float32)
        else {
          return
        }

        let maskProbsPointer = pointerWrapper.pointer + (4 + numClasses) * numAnchors + j
        let maskProbsData = maskProbs.dataPointer.assumingMemoryBound(to: Float.self)

        for i in 0..<maskConfidenceLength {
          maskProbsData[i] = maskProbsPointer[i * numAnchors]
        }

        let result = (boundingBox, Int(maxClassIndex), maxClassValue, maskProbs)

        resultsLock.lock()
        resultsWrapper.append(result)
        resultsLock.unlock()
      }
    }

    // Optimize NMS by grouping results by class first
    let results = resultsWrapper.getResults()
    var classBuckets: [Int: [(CGRect, Int, Float, MLMultiArray)]] = [:]
    for result in results {
      let classIndex = result.1
      if classBuckets[classIndex] == nil {
        classBuckets[classIndex] = []
        classBuckets[classIndex]!.reserveCapacity(results.count / numClasses + 1)
      }
      classBuckets[classIndex]!.append(result)
    }

    var selectedBoxesAndFeatures: [(CGRect, Int, Float, MLMultiArray)] = []
    selectedBoxesAndFeatures.reserveCapacity(results.count)

    for (_, classResults) in classBuckets {
      let boxesOnly = classResults.map { $0.0 }
      let scoresOnly = classResults.map { $0.2 }
      let selectedIndices = nonMaxSuppression(
        boxes: boxesOnly,
        scores: scoresOnly,
        threshold: iouThreshold
      )
      for idx in selectedIndices {
        selectedBoxesAndFeatures.append(classResults[idx])
      }
    }

    return selectedBoxesAndFeatures
  }

  func checkShapeDimensions(of multiArray: MLMultiArray) -> Int {
    let shapeAsInts = multiArray.shape.map { $0.intValue }
    let dimensionCount = shapeAsInts.count

    return dimensionCount
  }

}

final class FloatPointerWrapper: @unchecked Sendable {
  let pointer: UnsafeMutablePointer<Float>
  init(_ pointer: UnsafeMutablePointer<Float>) {
    self.pointer = pointer
  }
}

final class ResultsWrapper: @unchecked Sendable {
  private var results: [(CGRect, Int, Float, MLMultiArray)] = []

  func reserveCapacity(_ capacity: Int) {
    results.reserveCapacity(capacity)
  }

  func append(_ result: (CGRect, Int, Float, MLMultiArray)) {
    results.append(result)
  }

  func getResults() -> [(CGRect, Int, Float, MLMultiArray)] {
    return results
  }
}
