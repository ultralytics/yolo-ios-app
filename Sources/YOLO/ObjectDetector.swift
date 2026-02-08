// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, implementing object detection functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ObjectDetector class provides specialized functionality for detecting objects in images
//  using YOLO models. It processes Vision framework results to extract bounding boxes, class labels,
//  and confidence scores from model predictions. The class handles both real-time frame processing
//  and single image analysis, converting the Vision API's normalized coordinates to image coordinates,
//  and packaging the results in the standardized YOLOResult format. It includes performance monitoring
//  for inference time and frame rate, and offers runtime adjustable parameters such as confidence
//  threshold and IoU threshold for non-maximum suppression.

import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO object detection models that identifies and localizes objects in images.
///
/// This class processes the outputs from YOLO object detection models, extracting bounding boxes,
/// class labels, and confidence scores. It handles both real-time camera feed processing and
/// single image analysis, converting the normalized coordinates from the Vision framework
/// to image coordinates and applying non-maximum suppression to filter duplicative detections.
///
/// - Note: Object detection models output rectangular bounding boxes around detected objects.
/// - SeeAlso: `Segmenter` for models that produce pixel-level masks for objects.
public class ObjectDetector: BasePredictor, @unchecked Sendable {

  /// Sets the confidence threshold and updates the model's feature provider.
  ///
  /// This overridden method ensures that when the confidence threshold is changed,
  /// the Vision model's feature provider is also updated to use the new value.
  ///
  /// - Parameter confidence: The new confidence threshold value (0.0 to 1.0).
  override func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
    let iou = requiresNMS ? iouThreshold : 1.0
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iou, confidenceThreshold: confidenceThreshold)
  }

  /// Sets the IoU threshold and updates the model's feature provider.
  ///
  /// This overridden method ensures that when the IoU threshold is changed,
  /// the Vision model's feature provider is also updated to use the new value.
  ///
  /// - Parameter iou: The new IoU threshold value (0.0 to 1.0).
  override func setIouThreshold(iou: Double) {
    iouThreshold = iou
    let effectiveIou = requiresNMS ? iouThreshold : 1.0
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: effectiveIou, confidenceThreshold: confidenceThreshold)
  }

  /// Processes the results from the Vision framework's object detection request.
  ///
  /// This method extracts bounding boxes, class labels, and confidence scores from the
  /// Vision object detection results, converts coordinates to the original image space,
  /// and notifies listeners with the structured detection results.
  ///
  /// - Parameters:
  ///   - request: The completed Vision request containing object detection results.
  ///   - error: Any error that occurred during the Vision request.
  override func processObservations(for request: VNRequest, error: Error?) {
    var boxes = [Box]()

    // NMS-pipelined models (YOLO11 etc.) return VNRecognizedObjectObservation
    if let results = request.results as? [VNRecognizedObjectObservation] {
      for i in 0..<min(results.count, self.numItemsThreshold) {
        let prediction = results[i]
        let invertedBox = CGRect(
          x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
          width: prediction.boundingBox.width, height: prediction.boundingBox.height)
        let imageRect = VNImageRectForNormalizedRect(
          invertedBox, Int(inputSize.width), Int(inputSize.height))

        // The labels array is a list of VNClassificationObservation objects,
        // with the highest scoring class first in the list.
        let label = prediction.labels[0].identifier
        let index = self.labels.firstIndex(of: label) ?? 0
        let confidence = prediction.labels[0].confidence
        let box = Box(
          index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox)
        boxes.append(box)
      }
    }
    // NMS-free models (YOLO26) return raw MLMultiArray tensors
    else if let results = request.results as? [VNCoreMLFeatureValueObservation],
      let prediction = results.first?.featureValue.multiArrayValue
    {
      boxes = processRawResults(prediction)
    }

    // Measure FPS
    if self.t1 < 10.0 {  // valid dt
      self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
    }
    self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
    self.t3 = CACurrentMediaTime()

    self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)  // t2 seconds to ms
    let result = YOLOResult(
      orig_shape: inputSize, boxes: boxes, speed: self.t2, fps: 1 / self.t4, names: labels)

    self.currentOnResultsListener?.on(result: result)
  }

  /// Processes a static image and returns object detection results.
  ///
  /// This method performs object detection on a static image and returns the
  /// detection results synchronously. It handles the entire inference pipeline
  /// from setting up the Vision request to processing the detection results.
  ///
  /// - Parameter image: The CIImage to analyze for object detection.
  /// - Returns: A YOLOResult containing the detected objects with bounding boxes, class labels, and confidence scores.
  public override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
      return emptyResult
    }
    var boxes = [Box]()

    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    let start = Date()

    do {
      try requestHandler.perform([request])
      // NMS-pipelined models (YOLO11 etc.)
      if let results = request.results as? [VNRecognizedObjectObservation] {
        for i in 0..<min(results.count, self.numItemsThreshold) {
          let prediction = results[i]
          let invertedBox = CGRect(
            x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
            width: prediction.boundingBox.width, height: prediction.boundingBox.height)
          let imageRect = VNImageRectForNormalizedRect(
            invertedBox, Int(inputSize.width), Int(inputSize.height))

          // The labels array is a list of VNClassificationObservation objects,
          // with the highest scoring class first in the list.
          let label = prediction.labels[0].identifier
          let index = self.labels.firstIndex(of: label) ?? 0
          let confidence = prediction.labels[0].confidence
          let box = Box(
            index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox)
          boxes.append(box)
        }
      }
      // NMS-free models (YOLO26) return raw MLMultiArray tensors
      else if let results = request.results as? [VNCoreMLFeatureValueObservation],
        let prediction = results.first?.featureValue.multiArrayValue
      {
        boxes = processRawResults(prediction)
      }
    } catch {
      print(error)
    }
    _ = Date().timeIntervalSince(start)

    var result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: t1, names: labels)
    let annotatedImage = drawYOLODetections(on: image, result: result)
    result.annotatedImage = annotatedImage

    return result
  }

  // MARK: - Raw tensor processing (NMS-free YOLO26)

  /// Dispatches raw MLMultiArray tensor to the appropriate processing method based on output format.
  ///
  /// - Parameter prediction: The raw MLMultiArray output from the model.
  /// - Returns: An array of detected boxes.
  private func processRawResults(_ prediction: MLMultiArray) -> [Box] {
    let shape = prediction.shape.map { $0.intValue }
    let strides = prediction.strides.map { $0.intValue }
    let pointer = prediction.dataPointer.assumingMemoryBound(to: Float.self)
    let confThreshold = Float(confidenceThreshold)
    let modelW = CGFloat(modelInputSize.width)
    let modelH = CGFloat(modelInputSize.height)

    // Detect format: end2end [1, max_det, 6] vs traditional [1, 4+nc, num_anchors]
    guard shape.count == 3 else { return [] }
    let isEnd2End = shape[2] <= 6 || shape[2] < shape[1]

    if isEnd2End {
      return processEnd2EndResults(
        pointer: pointer, shape: shape, strides: strides,
        confThreshold: confThreshold, modelW: modelW, modelH: modelH)
    } else {
      return processTraditionalResults(
        pointer: pointer, shape: shape, strides: strides,
        confThreshold: confThreshold, modelW: modelW, modelH: modelH)
    }
  }

  /// Processes YOLO26 end2end output: [1, max_det, 6] = [x1, y1, x2, y2, conf, class_id] (xyxy pixel coords).
  ///
  /// - Parameters:
  ///   - pointer: Pointer to the raw float data.
  ///   - shape: The tensor shape [1, max_det, 6].
  ///   - strides: The tensor strides for correct indexing.
  ///   - confThreshold: Minimum confidence to include a detection.
  ///   - modelW: Model input width for coordinate normalization.
  ///   - modelH: Model input height for coordinate normalization.
  /// - Returns: An array of detected boxes.
  private func processEnd2EndResults(
    pointer: UnsafeMutablePointer<Float>, shape: [Int], strides: [Int],
    confThreshold: Float, modelW: CGFloat, modelH: CGFloat
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

      let normalizedBox = CGRect(
        x: x1 / modelW, y: y1 / modelH,
        width: (x2 - x1) / modelW, height: (y2 - y1) / modelH)
      let imageRect = VNImageRectForNormalizedRect(
        normalizedBox, Int(inputSize.width), Int(inputSize.height))
      let label = classIndex < labels.count ? labels[classIndex] : "\(classIndex)"

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
  ///   - modelW: Model input width for coordinate normalization.
  ///   - modelH: Model input height for coordinate normalization.
  /// - Returns: An array of detected boxes after non-maximum suppression.
  private func processTraditionalResults(
    pointer: UnsafeMutablePointer<Float>, shape: [Int], strides: [Int],
    confThreshold: Float, modelW: CGFloat, modelH: CGFloat
  ) -> [Box] {
    let numFeatures = shape[1]
    let numAnchors = shape[2]
    let numClasses = numFeatures - 4
    let iouThresh = Float(iouThreshold)
    let featureStride = strides[1]
    let anchorStride = strides[2]

    var candidateBoxes = [CGRect]()
    var candidateScores = [Float]()
    var candidateClasses = [Int]()

    for j in 0..<numAnchors {
      var bestScore: Float = 0
      var bestClass = 0
      for c in 0..<numClasses {
        let score = pointer[(4 + c) * featureStride + j * anchorStride]
        if score > bestScore { bestScore = score; bestClass = c }
      }
      guard bestScore > confThreshold else { continue }

      let x = pointer[j * anchorStride]
      let y = pointer[featureStride + j * anchorStride]
      let w = pointer[2 * featureStride + j * anchorStride]
      let h = pointer[3 * featureStride + j * anchorStride]
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
      let normalizedBox = CGRect(
        x: rect.minX / modelW, y: rect.minY / modelH,
        width: rect.width / modelW, height: rect.height / modelH)
      let imageRect = VNImageRectForNormalizedRect(
        normalizedBox, Int(inputSize.width), Int(inputSize.height))
      let classIndex = candidateClasses[i]
      let label = classIndex < labels.count ? labels[classIndex] : "\(classIndex)"
      boxes.append(
        Box(
          index: classIndex, cls: label, conf: candidateScores[i], xywh: imageRect,
          xywhn: normalizedBox))
    }
    return boxes
  }
}
