// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import Foundation
import Vision

/// Specialized predictor for YOLO object detection models.
public class ObjectDetector: BasePredictor {

  override func processResults() -> YOLOResult {
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
    }

    guard let results = request.results as? [VNRecognizedObjectObservation] else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
    }

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
}
