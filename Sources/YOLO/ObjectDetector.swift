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
  private var isYOLO26Model: Bool {
    guard let url = modelURL else {
      print("⚠️ ObjectDetector: No model URL available for YOLO26 detection")
      return false
    }
    
    // Check the full path and last path component for "yolo26"
    // This will match all YOLO26 sizes: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
    let fullPath = url.path.lowercased()
    let modelName = url.lastPathComponent.lowercased()
    
    // Remove file extensions to get base name
    let baseName = modelName
      .replacingOccurrences(of: ".mlmodelc", with: "")
      .replacingOccurrences(of: ".mlpackage", with: "")
      .replacingOccurrences(of: ".mlmodel", with: "")
    
    let isYOLO26 = fullPath.contains("yolo26") || baseName.contains("yolo26")
    
    if isYOLO26 {
      // Extract model size for logging
      let sizeMatch = baseName.range(of: "yolo26([nsmxl])", options: .regularExpression)
      let size = sizeMatch != nil ? String(baseName[sizeMatch!].dropFirst(5)) : "unknown"
      print("✅ ObjectDetector: Detected YOLO26\(size) model - skipping NMS")
    } else {
      print("ℹ️ ObjectDetector: Model appears to be YOLO11 or older - will apply NMS")
    }
    
    return isYOLO26
  }

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
      print("❌ ObjectDetector error: \(error.localizedDescription)")
      return
    }
    
    guard let results = request.results else {
      print("⚠️ ObjectDetector: No results from Vision request")
      return
    }
    
    print("🔍 ObjectDetector: Received \(results.count) results, type: \(type(of: results.first))")
    
    if let results = results as? [VNRecognizedObjectObservation] {
      print("✅ ObjectDetector: Found \(results.count) recognized object observations")
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
      print("📦 ObjectDetector: Created result with \(boxes.count) boxes")
    } else if let featureResults = results as? [VNCoreMLFeatureValueObservation] {
      print("✅ ObjectDetector: Handling model without built-in NMS (VNCoreMLFeatureValueObservation)")
      print("🔍 ObjectDetector: Model URL: \(modelURL?.path ?? "unknown")")
      print("🔍 ObjectDetector: Is YOLO26: \(isYOLO26Model)")
      // Handle models without built-in NMS - need manual post-processing
      guard let prediction = featureResults.first?.featureValue.multiArrayValue else {
        print("❌ ObjectDetector: No MLMultiArray in feature results")
        return
      }
      
      // Log raw prediction shape for debugging
      let rawShape = prediction.shape.map { $0.intValue }
      print("🔍 ObjectDetector: Raw prediction shape: \(rawShape)")
      
      // Post-process raw predictions
      let detectedObjects = postProcessDetection(
        feature: prediction,
        confidenceThreshold: Float(self.confidenceThreshold),
        iouThreshold: Float(self.iouThreshold)
      )
      
      print("🔍 ObjectDetector: Post-processed \(detectedObjects.count) detections")
      if !detectedObjects.isEmpty {
        let sample = detectedObjects.prefix(3)
        for (idx, det) in sample.enumerated() {
          let (box, classIdx, conf) = det
          let className = (classIdx < labels.count) ? labels[classIdx] : "unknown"
          print("  Detection \(idx): \(className) conf=\(conf) box=(\(box.minX), \(box.minY), \(box.width), \(box.height))")
        }
      }
      
      // Convert to Box format
      var boxes: [Box] = []
      let modelWidth = CGFloat(self.modelInputSize.width)
      let modelHeight = CGFloat(self.modelInputSize.height)
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
        
        // Debug: log the conversion
        if boxes.count < 2 {
          print("🔍 ObjectDetector: Converting box - normalized: \(rect), image coords: \(xywh), inputSize: \(inputWidth)x\(inputHeight)")
        }
        
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
      print("📦 ObjectDetector: Created result with \(boxes.count) boxes from raw predictions")
    } else {
      print("⚠️ ObjectDetector: Results are not VNRecognizedObjectObservation. Type: \(type(of: results.first))")
    }
  }
  
  /// Post-processes raw model output for detection models without built-in NMS
  private func postProcessDetection(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float)] {
    let shape = feature.shape.map { $0.intValue }
    
    print("🔍 ObjectDetector: Feature shape: \(shape), isYOLO26: \(isYOLO26Model)")
    
    // YOLO26 models might output in a different format:
    // - [1, num_detections, 6] or [num_detections, 6] where each row is [x, y, w, h, confidence, class] (post-NMS format)
    // - [batch, num_anchors, num_classes + 4] (anchor-based format like YOLO11)
    // - [num_anchors, num_classes + 4] (anchor-based format)
    
    // Check if this looks like YOLO26 post-NMS format: [1, num_detections, 6] or [num_detections, 6]
    if isYOLO26Model && shape.count >= 2 && shape.last == 6 {
      print("✅ ObjectDetector: Detected YOLO26 post-NMS format \(shape)")
      return postProcessYOLO26Format(feature: feature, shape: shape, confidenceThreshold: confidenceThreshold)
    }
    
    // YOLO detection output format: [batch, num_anchors, num_classes + 4] or [num_anchors, num_classes + 4]
    // Or sometimes: [1, num_anchors, num_classes + 4]
    var numAnchors: Int
    var numFeatures: Int
    
    if shape.count == 3 {
      // Format: [batch, num_anchors, features]
      numAnchors = shape[1]
      numFeatures = shape[2]
    } else if shape.count == 2 {
      // Format: [num_anchors, features]
      numAnchors = shape[0]
      numFeatures = shape[1]
    } else {
      print("❌ ObjectDetector: Unexpected feature shape: \(shape)")
      return []
    }
    
    let boxFeatureLength = 4  // x, y, w, h
    let numClasses = numFeatures - boxFeatureLength
    
    guard numClasses > 0 else {
      print("❌ ObjectDetector: Invalid number of classes: \(numClasses)")
      return []
    }
    
    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    
    // Extract detections
    var detections: [(CGRect, Int, Float)] = []
    detections.reserveCapacity(min(numAnchors / 10, 100))  // Estimate capacity
    
    // Helper function to apply sigmoid for YOLO26 (normalize logits to 0-1 range)
    func sigmoid(_ x: Float) -> Float {
      return 1.0 / (1.0 + exp(-x))
    }
    
    // Sample a few values to detect if scores need normalization
    var sampleScores: [Float] = []
    let sampleCount = min(10, numAnchors)
    for i in 0..<sampleCount {
      let offset = i * numFeatures
      for c in 0..<min(3, numClasses) {
        let score = featurePointer[offset + boxFeatureLength + c]
        sampleScores.append(score)
      }
    }
    let maxSample = sampleScores.max() ?? 0
    let minSample = sampleScores.min() ?? 0
    let needsNormalization = maxSample > 10.0 || minSample < -10.0  // Likely logits if outside 0-1 range
    
    if isYOLO26Model {
      print("🔍 ObjectDetector: YOLO26 model detected - sample scores range: [\(minSample), \(maxSample)]")
      if needsNormalization {
        print("⚠️ ObjectDetector: Scores appear to be logits - applying sigmoid normalization")
      }
    }
    
    for i in 0..<numAnchors {
      // Get box coordinates (normalized to model input size)
      // For [batch, anchors, features]: offset = batch_idx * anchors * features + anchor_idx * features
      // For [anchors, features]: offset = anchor_idx * features
      // Typically batch=1, so: offset = i * numFeatures
      let offset = i * numFeatures
      let cx = CGFloat(featurePointer[offset])
      let cy = CGFloat(featurePointer[offset + 1])
      let w = CGFloat(featurePointer[offset + 2])
      let h = CGFloat(featurePointer[offset + 3])
      
      // Convert center format to minX, minY format
      let boxX = cx - w / 2
      let boxY = cy - h / 2
      let box = CGRect(x: boxX, y: boxY, width: w, height: h)
      
      // Find best class
      var bestScore: Float = 0
      var bestClass: Int = 0
      for c in 0..<numClasses {
        var score = featurePointer[offset + boxFeatureLength + c]
        
        // For YOLO26 models, normalize scores appropriately
        if isYOLO26Model {
          // YOLO26 might output in different formats:
          // 1. Logits (very large positive/negative) - apply sigmoid
          // 2. 0-100 range - normalize to 0-1
          // 3. Already 0-1 - use as-is
          if abs(score) > 10.0 {
            // Likely logits, apply sigmoid
            score = sigmoid(score)
          } else if score > 1.0 && score <= 100.0 {
            // Likely 0-100 range, normalize to 0-1
            score = score / 100.0
          }
          // If already in 0-1 range, use as-is
        }
        
        if score > bestScore {
          bestScore = score
          bestClass = c
        }
      }
      
      // Normalize score to 0-1 range if needed (for threshold comparison)
      var normalizedScore = bestScore
      if bestScore > 1.0 && bestScore <= 100.0 {
        normalizedScore = bestScore / 100.0
      } else if abs(bestScore) > 10.0 {
        normalizedScore = sigmoid(bestScore)
      }
      
      // Apply confidence threshold (using normalized score)
      if normalizedScore > confidenceThreshold && normalizedScore <= 1.0 {
        detections.append((box, bestClass, normalizedScore))
      }
    }
    
    // YOLO26 models don't need NMS - they output final detections without NMS
    if isYOLO26Model {
      // For YOLO26, just sort by confidence and return (no NMS needed)
      print("🚫 ObjectDetector: Skipping NMS for YOLO26 model - returning \(detections.count) detections")
      detections.sort { $0.2 > $1.2 }
      return detections
    }
    
    print("🔧 ObjectDetector: Applying NMS for YOLO11/older model - \(detections.count) detections before NMS")
    
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
      print("❌ ObjectDetector: Invalid YOLO26 format, expected [1, num_detections, 6] or [num_detections, 6], got \(shape)")
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
    
    print("🔍 ObjectDetector: Processing YOLO26 format - model input size: \(modelWidth)x\(modelHeight), numDetections: \(numDetections)")
    
    // Sample first few detections to understand the coordinate format
    if numDetections > 0 {
      print("🔍 ObjectDetector: Confidence threshold: \(confidenceThreshold)")
      // Check first 10 detections for confidence distribution
      var maxConf: Float = 0
      var confidences: [Float] = []
      for i in 0..<min(10, numDetections) {
        let off = i * stride
        let conf = featurePointer[off + 4]
        confidences.append(conf)
        if conf > maxConf { maxConf = conf }
      }
      print("🔍 ObjectDetector: First 10 confidences: \(confidences.map { String(format: "%.4f", $0) })")
      print("🔍 ObjectDetector: Max confidence in first 10: \(maxConf)")
    }
    
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
      
      // Log first few detections for debugging
      if i < 3 {
        let className = classIndex < labels.count ? labels[classIndex] : "unknown"
        print("  Detection \(i): class=\(classIndex) '\(className)' conf=\(confidence) raw=(x1:\(x1), y1:\(y1), x2:\(x2), y2:\(y2)) normalized box=(\(boxX), \(boxY), \(boxW), \(boxH))")
      }
      
      // Validate: box should be reasonable size and within bounds
      let isValidBox = boxW > 0.01 && boxH > 0.01 && boxW <= 1.0 && boxH <= 1.0
      let hasValidConfidence = confidence > confidenceThreshold && confidence <= 1.0
      let hasValidClass = classIndex >= 0 && classIndex < labels.count
      
      if isValidBox && hasValidConfidence && hasValidClass {
        detections.append((box, classIndex, confidence))
      } else if i < 5 {
        // Log why detections are being filtered (only first few to avoid spam)
        if !isValidBox {
          print("  ⚠️ Detection \(i) filtered: invalid box size (w:\(boxW), h:\(boxH))")
        }
        if !hasValidConfidence {
          print("  ⚠️ Detection \(i) filtered: confidence \(confidence) below threshold \(confidenceThreshold)")
        }
        if !hasValidClass {
          print("  ⚠️ Detection \(i) filtered: invalid class index \(classIndex) (max: \(labels.count-1))")
        }
      }
    }
    
    // Sort by confidence (descending)
    detections.sort { $0.2 > $1.2 }
    
    print("✅ ObjectDetector: Processed \(detections.count) valid detections from YOLO26 format (out of \(numDetections) total)")
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
    _ = Date().timeIntervalSince(start)

    var result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: t1, names: labels)
    let annotatedImage = drawYOLODetections(on: image, result: result)
    result.annotatedImage = annotatedImage

    return result
  }
}

