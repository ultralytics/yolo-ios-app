// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Foundation
import Vision

/// Specialized predictor for YOLO object detection models.
public class ObjectDetector: BasePredictor {

  override func processResults() -> YOLOResult {
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
    }

    // NMS-pipelined models (legacy YOLO11 with nms=True) return VNRecognizedObjectObservation
    if let results = request.results as? [VNRecognizedObjectObservation] {
      return processNMSResults(results)
    }

    // NMS-free models (YOLO26 with nms=False) return raw MLMultiArray tensors
    if let results = request.results as? [VNCoreMLFeatureValueObservation],
      let prediction = results.first?.featureValue.multiArrayValue
    {
      return processRawResults(prediction)
    }

    return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
  }

  // MARK: - NMS-pipelined path (legacy YOLO11)

  private func processNMSResults(_ results: [VNRecognizedObjectObservation]) -> YOLOResult {
    var boxes = [Box]()
    for i in 0..<min(results.count, configuration.maxDetections) {
      let prediction = results[i]
      let invertedBox = CGRect(
        x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
        width: prediction.boundingBox.width, height: prediction.boundingBox.height)
      let imageRect = VNImageRectForNormalizedRect(
        invertedBox, Int(inputSize.width), Int(inputSize.height))
      let label = prediction.labels[0].identifier
      let index = self.labels.firstIndex(of: label) ?? 0
      let confidence = prediction.labels[0].confidence
      boxes.append(
        Box(index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox))
    }

    return YOLOResult(
      orig_shape: inputSize, boxes: boxes, speed: t1, fps: t4 > 0 ? 1 / t4 : nil, names: labels)
  }

  // MARK: - Raw tensor path (NMS-free YOLO26)

  private func processRawResults(_ prediction: MLMultiArray) -> YOLOResult {
    let shape = prediction.shape.map { $0.intValue }
    let strides = prediction.strides.map { $0.intValue }
    let pointer = prediction.dataPointer.assumingMemoryBound(to: Float.self)
    let confThreshold = Float(configuration.confidenceThreshold)
    let modelW = CGFloat(modelInputSize.width)
    let modelH = CGFloat(modelInputSize.height)

    // YOLO26 end2end detect output: [1, max_det, 6]
    // Each detection is [x1, y1, x2, y2, conf, class_id] in xyxy pixel coords
    // Traditional YOLO (non-end2end with nms=False): [1, 4+nc, num_anchors]
    let isEnd2End: Bool
    if shape.count == 3 {
      // End2end: last dim is 6 (or small), second dim is max_det (e.g. 300)
      // Traditional: second dim is 4+nc (e.g. 84), last dim is num_anchors (e.g. 8400)
      isEnd2End = shape[2] <= 6 || shape[2] < shape[1]
    } else {
      isEnd2End = false
    }

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

  /// YOLO26 end2end output: [1, max_det, 6] = [x1, y1, x2, y2, conf, class_id] (xyxy format)
  private func processEnd2EndResults(
    pointer: UnsafeMutablePointer<Float>, shape: [Int], strides: [Int],
    confThreshold: Float, modelW: CGFloat, modelH: CGFloat
  ) -> YOLOResult {
    let numDetections = shape[1]
    let numFields = shape[2]
    let detStride = strides[1]  // stride between detections
    let fieldStride = strides[2]  // stride between fields within a detection
    var boxes = [Box]()

    for i in 0..<numDetections {
      let base = i * detStride
      let conf = pointer[base + 4 * fieldStride]
      guard conf > confThreshold else { continue }

      // Coordinates are in xyxy pixel format relative to model input size
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

      if boxes.count >= configuration.maxDetections { break }
    }

    return YOLOResult(
      orig_shape: inputSize, boxes: boxes, speed: t1, fps: t4 > 0 ? 1 / t4 : nil, names: labels)
  }

  /// Traditional YOLO output: [1, 4+nc, num_anchors] — requires Swift NMS
  private func processTraditionalResults(
    pointer: UnsafeMutablePointer<Float>, shape: [Int], strides: [Int],
    confThreshold: Float, modelW: CGFloat, modelH: CGFloat
  ) -> YOLOResult {
    let numFeatures = shape[1]
    let numAnchors = shape[2]
    let numClasses = numFeatures - 4
    let iouThreshold = Float(configuration.iouThreshold)
    let featureStride = strides[1]
    let anchorStride = strides[2]

    var candidateBoxes = [CGRect]()
    var candidateScores = [Float]()
    var candidateClasses = [Int]()

    for j in 0..<numAnchors {
      var bestScore: Float = 0
      var bestClass: Int = 0
      for c in 0..<numClasses {
        let score = pointer[(4 + c) * featureStride + j * anchorStride]
        if score > bestScore {
          bestScore = score
          bestClass = c
        }
      }

      guard bestScore > confThreshold else { continue }

      let x = pointer[j * anchorStride]
      let y = pointer[featureStride + j * anchorStride]
      let w = pointer[2 * featureStride + j * anchorStride]
      let h = pointer[3 * featureStride + j * anchorStride]
      candidateBoxes.append(
        CGRect(
          x: CGFloat(x - w / 2), y: CGFloat(y - h / 2),
          width: CGFloat(w), height: CGFloat(h)))
      candidateScores.append(bestScore)
      candidateClasses.append(bestClass)
    }

    let selectedIndices = nonMaxSuppression(
      boxes: candidateBoxes, scores: candidateScores, threshold: iouThreshold)

    var boxes = [Box]()
    for i in selectedIndices.prefix(configuration.maxDetections) {
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
          index: classIndex, cls: label, conf: candidateScores[i],
          xywh: imageRect, xywhn: normalizedBox))
    }

    return YOLOResult(
      orig_shape: inputSize, boxes: boxes, speed: t1, fps: t4 > 0 ? 1 / t4 : nil, names: labels)
  }
}
