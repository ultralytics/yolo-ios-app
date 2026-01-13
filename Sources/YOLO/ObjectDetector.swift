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

  /// Checks if the current model is a YOLO26 model (which doesn't need NMS)
  /// Works for all YOLO26 sizes: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
  private var isYOLO26Model: Bool { isYOLO26Model(from: modelURL) }

  /// Sets the confidence threshold and updates the model's feature provider.
  ///
  /// This overridden method ensures that when the confidence threshold is changed,
  /// the Vision model's feature provider is also updated to use the new value.
  ///
  /// - Parameter confidence: The new confidence threshold value (0.0 to 1.0).
  override func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  /// Sets the IoU threshold and updates the model's feature provider.
  ///
  /// This overridden method ensures that when the IoU threshold is changed,
  /// the Vision model's feature provider is also updated to use the new value.
  ///
  /// - Parameter iou: The new IoU threshold value (0.0 to 1.0).
  override func setIouThreshold(iou: Double) {
    iouThreshold = iou
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
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
    if let error = error {
      print("ObjectDetector error: \(error.localizedDescription)")
      return
    }

    guard let results = request.results else {
      return
    }

    if let results = results as? [VNRecognizedObjectObservation] {
      var boxes = [Box]()

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

      // Measure FPS
      if self.t1 < 10.0 {  // valid dt
        self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
      }
      self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
      self.t3 = CACurrentMediaTime()

      self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)  // t2 seconds to ms
      //                self.currentOnFpsRateListener?.on(fpsRate: 1 / self.t4)
      let result = YOLOResult(
        orig_shape: inputSize, boxes: boxes, speed: self.t2, fps: 1 / self.t4, names: labels)

      self.currentOnResultsListener?.on(result: result)
    } else if let featureResults = results as? [VNCoreMLFeatureValueObservation] {
      // Handle models without built-in NMS - need manual post-processing
      guard let prediction = featureResults.first?.featureValue.multiArrayValue else {
        print("ObjectDetector: No MLMultiArray in feature results")
        return
      }

      // Post-process raw predictions
      let detectedObjects = postProcessDetection(
        feature: prediction,
        confidenceThreshold: Float(self.confidenceThreshold),
        iouThreshold: Float(self.iouThreshold)
      )

      // Convert to Box format
      var boxes: [Box] = []
      let inputWidth = Int(inputSize.width)
      let inputHeight = Int(inputSize.height)

      let limitedObjects = detectedObjects.prefix(self.numItemsThreshold)
      for detection in limitedObjects {
        let (box, classIndex, confidence) = detection
        // Box coordinates from postProcessYOLO26Format are already normalized (0-1 range)
        // So we can use them directly without dividing by model size
        let rect = CGRect(
          x: box.minX,
          y: box.minY,
          width: box.width,
          height: box.height
        )
        let label = (classIndex < labels.count) ? labels[classIndex] : "unknown"
        let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

        let boxResult = Box(
          index: classIndex,
          cls: label,
          conf: confidence,
          xywh: xywh,
          xywhn: rect
        )
        boxes.append(boxResult)
      }

      // Measure FPS
      if self.t1 < 10.0 {
        self.t2 = self.t1 * 0.05 + self.t2 * 0.95
      }
      self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95
      self.t3 = CACurrentMediaTime()

      let result = YOLOResult(
        orig_shape: inputSize,
        boxes: boxes,
        speed: self.t2,
        fps: 1 / self.t4,
        names: labels
      )

      self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)
      self.currentOnResultsListener?.on(result: result)
    }
  }

  /// Post-processes raw model output for detection models without built-in NMS
  private func postProcessDetection(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float)] {
    let shape = feature.shape.map { $0.intValue }

    // YOLO26 models might output in a different format:
    // - [1, num_detections, 6] or [num_detections, 6] where each row is [x, y, w, h, confidence, class] (post-NMS format)
    // - [batch, num_anchors, num_classes + 4] (anchor-based format like YOLO11)
    // - [num_anchors, num_classes + 4] (anchor-based format)

    // Check if this looks like YOLO26 post-NMS format: [1, num_detections, 6] or [num_detections, 6]
    if isYOLO26Model && shape.count >= 2 && shape.last == 6 {
      return postProcessYOLO26Format(
        feature: feature, shape: shape, confidenceThreshold: confidenceThreshold)
    }

    // YOLO detection output format: [batch, num_anchors, num_classes + 4] (anchor-first)
    // or [batch, num_features, num_anchors] (feature-first, e.g., 1x84x8400).
    var numAnchors: Int
    var numFeatures: Int
    enum FeatureLayout { case anchorFirst, featureFirst }
    let layout: FeatureLayout

    if shape.count == 3 {
      if shape[1] > shape[2] {
        layout = .anchorFirst
        numAnchors = shape[1]
        numFeatures = shape[2]
      } else {
        layout = .featureFirst
        numFeatures = shape[1]
        numAnchors = shape[2]
      }
    } else if shape.count == 2 {
      layout = .anchorFirst
      numAnchors = shape[0]
      numFeatures = shape[1]
    } else {
      print("ObjectDetector: Unexpected feature shape: \(shape)")
      return []
    }

    let boxFeatureLength = 4  // x, y, w, h
    let numClasses = numFeatures - boxFeatureLength

    guard numClasses > 0 else {
      print("ObjectDetector: Invalid number of classes: \(numClasses)")
      return []
    }

    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)

    func value(featureIndex: Int, anchorIndex: Int) -> Float {
      switch layout {
      case .anchorFirst:
        return featurePointer[anchorIndex * numFeatures + featureIndex]
      case .featureFirst:
        return featurePointer[featureIndex * numAnchors + anchorIndex]
      }
    }

    // Extract detections
    var detections: [(CGRect, Int, Float)] = []
    detections.reserveCapacity(min(numAnchors / 10, 100))  // Estimate capacity

    for i in 0..<numAnchors {
      // Get box coordinates (normalized to model input size)
      // For [batch, anchors, features]: offset = batch_idx * anchors * features + anchor_idx * features
      // For [anchors, features]: offset = anchor_idx * features
      // Typically batch=1, so: offset = i * numFeatures
      let cx = CGFloat(value(featureIndex: 0, anchorIndex: i))
      let cy = CGFloat(value(featureIndex: 1, anchorIndex: i))
      let w = CGFloat(value(featureIndex: 2, anchorIndex: i))
      let h = CGFloat(value(featureIndex: 3, anchorIndex: i))

      // Convert center format to minX, minY format
      let boxX = cx - w / 2
      let boxY = cy - h / 2
      let box = CGRect(x: boxX, y: boxY, width: w, height: h)

      // Find best class
      var bestScore: Float = 0
      var bestClass: Int = 0
      for c in 0..<numClasses {
        var score = value(featureIndex: boxFeatureLength + c, anchorIndex: i)
        if isYOLO26Model {
          score = normalizeYOLO26Score(score)
        }
        if score > bestScore {
          bestScore = score
          bestClass = c
        }
      }

      // Normalize score to 0-1 range if needed (for threshold comparison)
      let normalizedScore = isYOLO26Model ? normalizeYOLO26Score(bestScore) : bestScore

      // Apply confidence threshold (using normalized score)
      if normalizedScore > confidenceThreshold && normalizedScore <= 1.0 {
        detections.append((box, bestClass, normalizedScore))
      }
    }

    // YOLO26 models don't need NMS - they output final detections without NMS
    if isYOLO26Model {
      // For YOLO26, just sort by confidence and return (no NMS needed)
      detections.sort { $0.2 > $1.2 }
      return detections
    }

    // For YOLO11 and older models, apply NMS per class
    var classBuckets: [Int: [(CGRect, Int, Float)]] = [:]
    for detection in detections {
      let classIndex = detection.1
      if classBuckets[classIndex] == nil {
        classBuckets[classIndex] = []
      }
      classBuckets[classIndex]?.append(detection)
    }

    var selectedDetections: [(CGRect, Int, Float)] = []
    for (_, classDetections) in classBuckets {
      let boxesOnly = classDetections.map { $0.0 }
      let scoresOnly = classDetections.map { $0.2 }
      let selectedIndices = nonMaxSuppression(
        boxes: boxesOnly,
        scores: scoresOnly,
        threshold: iouThreshold
      )
      for idx in selectedIndices {
        selectedDetections.append(classDetections[idx])
      }
    }

    // Sort by confidence (descending)
    selectedDetections.sort { $0.2 > $1.2 }

    return selectedDetections
  }

  /// Post-processes YOLO26 format: [1, num_detections, 6] or [num_detections, 6] where each row is [x, y, w, h, confidence, class]
  private func postProcessYOLO26Format(
    feature: MLMultiArray,
    shape: [Int],
    confidenceThreshold: Float
  ) -> [(CGRect, Int, Float)] {
    // Handle both [1, num_detections, 6] and [num_detections, 6] formats
    let numDetections: Int
    if shape.count == 3 {
      numDetections = shape[1]  // [batch, num_detections, 6]
    } else if shape.count == 2 {
      numDetections = shape[0]  // [num_detections, 6]
    } else {
      print(
        "ObjectDetector: Invalid YOLO26 format, expected [1, num_detections, 6] or [num_detections, 6], got \(shape)"
      )
      return []
    }

    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    var detections: [(CGRect, Int, Float)] = []

    // Calculate stride based on shape
    let stride: Int
    if shape.count == 3 {
      stride = shape[2]  // Skip batch dimension: [batch, detections, features]
    } else {
      stride = shape[1]  // [detections, features]
    }

    let modelWidth = CGFloat(self.modelInputSize.width)
    let modelHeight = CGFloat(self.modelInputSize.height)

    for i in 0..<numDetections {
      // For [1, 300, 6] format, data is stored as: [batch][detection][feature]
      // MLMultiArray uses row-major order, so offset = i * 6 for detection i
      let offset = i * stride

      // YOLO26 format: Based on Ultralytics export, YOLO26 outputs [x1, y1, x2, y2, confidence, class]
      // in pixel coordinates (corner format, not center format)
      let x1 = CGFloat(featurePointer[offset])
      let y1 = CGFloat(featurePointer[offset + 1])
      let x2 = CGFloat(featurePointer[offset + 2])
      let y2 = CGFloat(featurePointer[offset + 3])
      var confidence = featurePointer[offset + 4]
      let classIndex = Int(round(featurePointer[offset + 5]))

      // Normalize confidence: YOLO26 outputs in 0-1 range (already normalized)
      // But check if it's in 0-100 range
      if confidence > 1.0 && confidence <= 100.0 {
        confidence = confidence / 100.0
      } else if confidence > 100.0 {
        // If > 100, might be logits - apply sigmoid
        confidence = 1.0 / (1.0 + exp(-confidence))
      }
      // If already 0-1, use as-is

      // Convert corner coordinates [x1, y1, x2, y2] from pixel space to normalized 0-1
      var boxX: CGFloat = 0
      var boxY: CGFloat = 0
      var boxW: CGFloat = 0
      var boxH: CGFloat = 0

      if modelWidth > 0 && modelHeight > 0 {
        // Normalize from pixel coordinates to 0-1 range
        boxX = x1 / modelWidth
        boxY = y1 / modelHeight
        boxW = (x2 - x1) / modelWidth
        boxH = (y2 - y1) / modelHeight
      } else {
        // If model size unknown, assume already normalized
        boxX = x1
        boxY = y1
        boxW = x2 - x1
        boxH = y2 - y1
      }

      // Clamp to valid 0-1 range
      boxX = max(0.0, min(1.0, boxX))
      boxY = max(0.0, min(1.0, boxY))
      boxW = max(0.0, min(1.0 - boxX, boxW))
      boxH = max(0.0, min(1.0 - boxY, boxH))

      let box = CGRect(x: boxX, y: boxY, width: boxW, height: boxH)

      // Validate: box should be reasonable size and within bounds
      let isValidBox = boxW > 0.01 && boxH > 0.01 && boxW <= 1.0 && boxH <= 1.0
      let hasValidConfidence = confidence > confidenceThreshold && confidence <= 1.0
      let hasValidClass = classIndex >= 0 && classIndex < labels.count

      if isValidBox && hasValidConfidence && hasValidClass {
        detections.append((box, classIndex, confidence))
      }
    }

    // Sort by confidence (descending)
    detections.sort { $0.2 > $1.2 }

    return detections
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
    } catch {
      print(error)
    }

    var result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: t1, names: labels)
    let annotatedImage = drawYOLODetections(on: image, result: result)
    result.annotatedImage = annotatedImage

    return result
  }
}
