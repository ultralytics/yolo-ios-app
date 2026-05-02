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
public final class Segmenter: BasePredictor, @unchecked Sendable {

  override func processObservations(for request: VNRequest, _ error: Error?) {
    guard let parsed = parseSegmentationRequest(request) else {
      self.isUpdating = false
      return
    }
    let limitedObjects = Array(parsed.detectedObjects.prefix(self.numItemsThreshold))
    let boxes = buildBoxes(from: limitedObjects)

    // Update timing before capturing values to avoid one-frame lag.
    self.updateTime()

    let capturedMasks = parsed.masks
    let capturedInputSize = self.inputSize
    let capturedModelInputSize = self.modelInputSize
    let capturedT2 = self.t2
    let capturedT4 = self.t4
    let capturedLabels = self.labels

    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
      guard
        let processed = generateCombinedMaskImage(
          detectedObjects: limitedObjects,
          protos: capturedMasks,
          inputWidth: capturedModelInputSize.width,
          inputHeight: capturedModelInputSize.height,
          threshold: 0.5
        ) as? (CGImage?, [[[Float]]])
      else {
        DispatchQueue.main.async { [weak self] in self?.isUpdating = false }
        return
      }
      let result = YOLOResult(
        orig_shape: capturedInputSize,
        boxes: boxes,
        masks: Masks(masks: processed.1, combinedMask: processed.0),
        speed: capturedT2, fps: 1 / capturedT4, names: capturedLabels)
      self?.currentOnResultsListener?.on(result: result)
    }
  }

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    self.inputSize = CGSize(width: image.extent.width, height: image.extent.height)
    var result = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)

    do {
      try requestHandler.perform([request])
      guard let parsed = parseSegmentationRequest(request) else { return result }

      let limitedObjects = Array(parsed.detectedObjects.prefix(self.numItemsThreshold))
      let boxes = buildBoxes(from: limitedObjects)

      guard
        let processed = generateCombinedMaskImage(
          detectedObjects: limitedObjects, protos: parsed.masks,
          inputWidth: self.modelInputSize.width, inputHeight: self.modelInputSize.height,
          threshold: 0.5
        ) as? (CGImage?, [[[Float]]])
      else {
        return YOLOResult(orig_shape: inputSize, boxes: boxes, speed: 0, names: labels)
      }

      updateTime()
      result = YOLOResult(
        orig_shape: inputSize, boxes: boxes,
        masks: Masks(masks: processed.1, combinedMask: processed.0),
        annotatedImage: drawYOLOSegmentationWithBoxes(
          ciImage: image, boxes: boxes, maskImage: processed.0),
        speed: self.t2, fps: 1 / self.t4, names: labels)
    } catch {
      YOLOLog.error("Segmentation failed: \(error)")
    }
    return result
  }

  /// Pulls the `(pred, masks, detectedObjects)` tuple out of a completed Vision request.
  /// The shape-4 output is the prototype masks and the other is the detection features;
  /// their order depends on the exported model.
  private func parseSegmentationRequest(
    _ request: VNRequest
  ) -> (masks: MLMultiArray, detectedObjects: [(CGRect, Int, Float, MLMultiArray)])? {
    guard let results = request.results as? [VNCoreMLFeatureValueObservation],
      results.count == 2,
      let out0 = results[0].featureValue.multiArrayValue,
      let out1 = results[1].featureValue.multiArrayValue
    else { return nil }

    let (masks, pred): (MLMultiArray, MLMultiArray) =
      checkShapeDimensions(of: out0) == 4 ? (out0, out1) : (out1, out0)
    let detectedObjects = postProcessSegment(
      feature: pred,
      confidenceThreshold: Float(confidenceThreshold),
      iouThreshold: Float(iouThreshold))
    return (masks, detectedObjects)
  }

  /// Converts post-processed (rect, class, score, maskCoeffs) tuples into `Box` values
  /// mapped from model-space to the current input-image coordinate space.
  private func buildBoxes(from objects: [(CGRect, Int, Float, MLMultiArray)]) -> [Box] {
    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)
    let inputWidth = Int(inputSize.width)
    let inputHeight = Int(inputSize.height)
    var boxes: [Box] = []
    boxes.reserveCapacity(objects.count)
    for (box, classIndex, confidence, _) in objects {
      guard classIndex < labels.count else { continue }
      let rect = CGRect(
        x: box.minX / modelWidth, y: box.minY / modelHeight,
        width: box.width / modelWidth, height: box.height / modelHeight)
      let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)
      boxes.append(
        Box(
          index: classIndex, cls: labels[classIndex], conf: confidence,
          xywh: xywh, xywhn: rect))
    }
    return boxes
  }

  nonisolated func postProcessSegment(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let shape = feature.shape.map { $0.intValue }
    guard shape.count == 3 else { return [] }

    // YOLO26 end2end seg: [1, max_det, 6+32] where shape[2] < shape[1]
    // Traditional seg: [1, 4+nc+32, num_anchors] where shape[2] > shape[1]
    if shape[2] < shape[1] {
      return postProcessEnd2EndSegment(
        feature: feature, shape: shape, confidenceThreshold: confidenceThreshold)
    }

    let numAnchors = shape[2]
    let numFeatures = shape[1]
    let boxFeatureLength = 4
    let maskConfidenceLength = 32
    let numClasses = numFeatures - boxFeatureLength - maskConfidenceLength

    // Wrapper for thread-safe results collection
    final class ResultsWrapper: @unchecked Sendable {
      private let lock = NSLock()
      private(set) var results: [(CGRect, Int, Float, MLMultiArray)] = []

      func append(_ result: (CGRect, Int, Float, MLMultiArray)) {
        lock.lock()
        results.append(result)
        lock.unlock()
      }
    }

    let resultsWrapper = ResultsWrapper()

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

        resultsWrapper.append(result)

      }
    }

    let collectedResults = resultsWrapper.results

    // Group results by class for per-class NMS
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

  /// Processes YOLO26 end2end segmentation output: [1, max_det, 6+32].
  /// Each detection: [x1, y1, x2, y2, conf, class_id, mask_0...mask_31] in xyxy pixel coords.
  /// NMS is already applied by the model, so no additional NMS is needed.
  private nonisolated func postProcessEnd2EndSegment(
    feature: MLMultiArray,
    shape: [Int],
    confidenceThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let numDetections = shape[1]
    let numFields = shape[2]
    let maskCoefficients = 32
    let strides = feature.strides.map { $0.intValue }
    let pointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let detStride = strides[1]
    let fieldStride = strides[2]

    var results: [(CGRect, Int, Float, MLMultiArray)] = []

    for i in 0..<numDetections {
      let base = i * detStride
      let conf = pointer[base + 4 * fieldStride]
      guard conf > confidenceThreshold else { continue }

      let x1 = CGFloat(pointer[base])
      let y1 = CGFloat(pointer[base + fieldStride])
      let x2 = CGFloat(pointer[base + 2 * fieldStride])
      let y2 = CGFloat(pointer[base + 3 * fieldStride])
      let classId = numFields > 5 ? Int(pointer[base + 5 * fieldStride]) : 0

      let boundingBox = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)

      // Extract mask coefficients (fields 6..37)
      guard
        let maskProbs = try? MLMultiArray(
          shape: [NSNumber(value: maskCoefficients)], dataType: .float32)
      else { continue }
      let maskProbsData = maskProbs.dataPointer.assumingMemoryBound(to: Float.self)
      let maskStartField = numFields > 5 ? 6 : 5
      for m in 0..<min(maskCoefficients, numFields - maskStartField) {
        maskProbsData[m] = pointer[base + (maskStartField + m) * fieldStride]
      }

      results.append((boundingBox, classId, conf, maskProbs))
    }

    return results
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
