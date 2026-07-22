// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO SDK, implementing object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  ObjectDetector extracts bounding boxes, class labels, and confidence scores from YOLO detection model outputs,
//  handling both real-time camera frames and single-image analysis. It supports the traditional Vision NMS pipeline
//  (YOLO11, returning `VNRecognizedObjectObservation`) and NMS-free end2end models (YOLO26, returning raw
//  `MLMultiArray` tensors), converting model coordinates to input-image space and packaging results as `YOLOResult`.

import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO object detection models that identifies and localizes objects in images.
///
/// Handles both real-time camera frames and single-image analysis, converting Vision-normalized coordinates to
/// image-space rectangles. Traditional models pass through the Vision NMS pipeline; NMS-free YOLO26 models are
/// decoded directly from raw `MLMultiArray` tensors with optional Swift-side NMS.
///
/// - SeeAlso: `Segmenter` for models that also produce pixel-level masks.
public final class ObjectDetector: BasePredictor, @unchecked Sendable {

  /// Processes the results from the Vision framework's object detection request.
  ///
  /// Decodes boxes from `request.results`, converts them to image-space coordinates, and notifies the results
  /// listener.
  ///
  /// - Parameters:
  ///   - request: The completed Vision request containing object detection results.
  ///   - error: Any error that occurred during the Vision request.
  override func processObservations(for request: VNRequest, _ error: Error?) {
    markInferenceEnd()
    let boxes = decodeBoxes(from: request)
    self.updateTime()
    var result = YOLOResult(
      orig_shape: inputSize, boxes: boxes, speed: self.t2, fps: 1 / self.t4, names: labels)
    applyTimingBreakdown(&result, smoothed: true)
    result.originalImage = currentOriginalImage
    self.currentOnResultsListener?.on(result: result)
  }

  /// Decodes detection boxes from a completed Vision request.
  ///
  /// Handles both NMS-pipelined models (`VNRecognizedObjectObservation`, e.g. YOLO11) and NMS-free models
  /// (`VNCoreMLFeatureValueObservation` raw tensors, e.g. YOLO26).
  private func decodeBoxes(from request: VNRequest) -> [Box] {
    // NMS-pipelined models (YOLO11 etc.) return VNRecognizedObjectObservation
    if let results = request.results as? [VNRecognizedObjectObservation] {
      var boxes = [Box]()
      boxes.reserveCapacity(min(results.count, self.numItemsThreshold))
      for i in 0..<min(results.count, self.numItemsThreshold) {
        let prediction = results[i]
        let invertedBox = CGRect(
          x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
          width: prediction.boundingBox.width, height: prediction.boundingBox.height)
        let imageRect = VNImageRectForNormalizedRect(
          invertedBox, Int(inputSize.width), Int(inputSize.height))

        // The labels array is a list of VNClassificationObservation objects, with the highest scoring class first in
        // the list.
        let observationLabel = prediction.labels.first?.identifier ?? ""
        let index = self.labels.firstIndex(of: observationLabel) ?? 0
        let label = observationLabel.isEmpty ? labelName(for: index) : observationLabel
        let confidence = prediction.labels.first?.confidence ?? 0
        boxes.append(
          Box(index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox))
      }
      return boxes
    }
    // NMS-free models (YOLO26) return raw MLMultiArray tensors
    if let results = request.results as? [VNCoreMLFeatureValueObservation],
      let prediction = results.first?.featureValue.multiArrayValue
    {
      return processRawResults(prediction)
    }
    return []
  }

  /// Runs synchronous object detection on a static image and returns the results.
  ///
  /// - Parameter image: The CIImage to analyze for object detection.
  /// - Returns: A YOLOResult containing the detected objects with bounding boxes, class labels, and confidence scores.
  public override func predictOnImage(image: CIImage) -> YOLOResult {
    guard let request = visionRequest else {
      let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
      return emptyResult
    }
    var boxes = [Box]()

    let requestHandler = makeRequestHandler(for: image)
    if perform(request, with: requestHandler, errorMessage: "Object detection failed") {
      markInferenceEnd()
      boxes = decodeBoxes(from: request)
    }

    var result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: 0, names: labels)
    result.speed = finishTiming(notify: false)  // before drawing: annotation is excluded from timings
    applyTimingBreakdown(&result)
    let annotatedImage = drawYOLODetections(on: image, result: result)
    result.annotatedImage = annotatedImage
    if capturesOriginalImage {
      result.originalImage = UIImage(ciImage: image)
    }

    return result
  }

  // MARK: - Raw tensor processing (NMS-free YOLO26)

  /// Dispatches a raw `MLMultiArray` tensor to the appropriate decoder based on output shape (end2end vs traditional).
  ///
  /// - Parameter prediction: The raw MLMultiArray output from the model.
  /// - Returns: An array of detected boxes.
  func processRawResults(_ prediction: MLMultiArray) -> [Box] {
    let shape = prediction.shape.map { $0.intValue }
    let strides = prediction.strides.map { $0.intValue }
    let confThreshold = Float(confidenceThreshold)

    // Detect format: end2end [1, max_det, 6] vs traditional [1, 4+nc, num_anchors]
    guard shape.count == 3, strides.count == 3 else { return [] }
    let isEnd2End = shape[2] < shape[1]

    let values = Self.floatValues(from: prediction)
    return values.withUnsafeBufferPointer { buffer in
      guard let pointer = buffer.baseAddress else { return [] }
      if isEnd2End {
        return processEnd2EndResults(
          pointer: pointer, shape: shape, strides: strides,
          confThreshold: confThreshold)
      } else {
        return processTraditionalResults(
          pointer: pointer, shape: shape, strides: strides,
          confThreshold: confThreshold)
      }
    }
  }

  /// Copies a raw model output tensor into a contiguous `Float` buffer.
  ///
  /// Uses a fast direct memory copy when the tensor is already `.float32`, matching its natural storage, and falls
  /// back to `MLMultiArray`'s safe per-element accessor for any other backing type (e.g. `.float16`, which some
  /// ANE-optimized NMS-free exports use for their raw detection output). Reinterpreting a non-`.float32` buffer as
  /// `Float` via `assumingMemoryBound` would silently read the wrong byte offsets and run past the tensor's actual
  /// allocation as detections accumulate.
  private static func floatValues(from multiArray: MLMultiArray) -> [Float] {
    let count = multiArray.count
    var values = [Float](repeating: 0, count: count)
    if multiArray.dataType == .float32 {
      let source = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
      values.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: source, count: count) }
    } else {
      for i in 0..<count { values[i] = multiArray[i].floatValue }
    }
    return values
  }

  /// Processes YOLO26 end2end output: [1, max_det, 6] = [x1, y1, x2, y2, conf, class_id] (xyxy pixel coords).
  ///
  /// - Parameters:
  ///   - pointer: Pointer to the raw float data.
  ///   - shape: The tensor shape [1, max_det, 6].
  ///   - strides: The tensor strides for correct indexing.
  ///   - confThreshold: Minimum confidence to include a detection.
  /// - Returns: An array of detected boxes.
  private func processEnd2EndResults(
    pointer: UnsafePointer<Float>, shape: [Int], strides: [Int],
    confThreshold: Float
  ) -> [Box] {
    let numDetections = shape[1]
    let numFields = shape[2]
    let detStride = strides[1]
    let fieldStride = strides[2]
    var boxes = [Box]()

    for i in 0..<numDetections {
      let base = i * detStride
      let conf = pointer[base + 4 * fieldStride]
      guard conf > confThreshold else { continue }

      let x1 = CGFloat(pointer[base])
      let y1 = CGFloat(pointer[base + fieldStride])
      let x2 = CGFloat(pointer[base + 2 * fieldStride])
      let y2 = CGFloat(pointer[base + 3 * fieldStride])
      let classIndex = numFields > 5 ? Int(pointer[base + 5 * fieldStride]) : 0

      let imageRect = inputRect(
        fromModelRect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1))
      let normalizedBox = normalizedRect(fromInputRect: imageRect)
      let label = labelName(for: classIndex)

      boxes.append(
        Box(index: classIndex, cls: label, conf: conf, xywh: imageRect, xywhn: normalizedBox))
      if boxes.count >= numItemsThreshold { break }
    }
    return boxes
  }

  /// Processes traditional YOLO output: [1, 4+nc, num_anchors] in xywh format, requiring Swift NMS.
  ///
  /// - Parameters:
  ///   - pointer: Pointer to the raw float data.
  ///   - shape: The tensor shape [1, 4+nc, num_anchors].
  ///   - strides: The tensor strides for correct indexing.
  ///   - confThreshold: Minimum confidence to include a detection.
  /// - Returns: An array of detected boxes after non-maximum suppression.
  private func processTraditionalResults(
    pointer: UnsafePointer<Float>, shape: [Int], strides: [Int],
    confThreshold: Float
  ) -> [Box] {
    let numFeatures = shape[1]
    let numAnchors = shape[2]
    let numClasses = numFeatures - 4
    guard numClasses > 0 else { return [] }
    let iouThresh = Float(iouThreshold)
    let featureStride = strides[1]
    let anchorStride = strides[2]

    var candidateBoxes = [CGRect]()
    var candidateScores = [Float]()
    var candidateClasses = [Int]()

    for j in 0..<numAnchors {
      let anchorOffset = j * anchorStride
      var bestScore: Float = 0
      var bestClass = 0
      for c in 0..<numClasses {
        let score = pointer[(4 + c) * featureStride + anchorOffset]
        if score > bestScore {
          bestScore = score
          bestClass = c
        }
      }
      guard bestScore > confThreshold else { continue }

      let x = pointer[anchorOffset]
      let y = pointer[featureStride + anchorOffset]
      let w = pointer[2 * featureStride + anchorOffset]
      let h = pointer[3 * featureStride + anchorOffset]
      candidateBoxes.append(
        CGRect(x: CGFloat(x - w / 2), y: CGFloat(y - h / 2), width: CGFloat(w), height: CGFloat(h))
      )
      candidateScores.append(bestScore)
      candidateClasses.append(bestClass)
    }

    let selectedIndices = nonMaxSuppression(
      boxes: candidateBoxes, scores: candidateScores, threshold: iouThresh)

    var boxes = [Box]()
    for i in selectedIndices.prefix(numItemsThreshold) {
      let rect = candidateBoxes[i]
      let imageRect = inputRect(fromModelRect: rect)
      let normalizedBox = normalizedRect(fromInputRect: imageRect)
      let classIndex = candidateClasses[i]
      let label = labelName(for: classIndex)
      boxes.append(
        Box(
          index: classIndex, cls: label, conf: candidateScores[i], xywh: imageRect,
          xywhn: normalizedBox))
    }
    return boxes
  }
}
