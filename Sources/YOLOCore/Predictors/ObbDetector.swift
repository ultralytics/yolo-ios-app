// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Accelerate
import CoreML
import Foundation
import Vision

/// Specialized predictor for YOLO oriented bounding box detection models.
public class ObbDetector: BasePredictor {

  override func processResults() -> YOLOResult {
    guard let request = visionRequest,
      let results = request.results as? [VNCoreMLFeatureValueObservation],
      let prediction = results.first?.featureValue.multiArrayValue
    else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
    }

    let nmsResults = postProcessOBB(
      feature: prediction,
      confidenceThreshold: Float(configuration.confidenceThreshold),
      iouThreshold: Float(configuration.iouThreshold))

    var obbResults = [OBBResult]()
    for result in nmsResults.prefix(configuration.maxDetections) {
      obbResults.append(
        OBBResult(
          box: result.box, confidence: result.score,
          cls: labels[result.cls], index: result.cls))
    }

    return YOLOResult(
      orig_shape: inputSize, boxes: [], obb: obbResults,
      speed: t1, fps: t4 > 0 ? 1 / t4 : nil, names: labels)
  }

  private func postProcessOBB(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(box: OBB, score: Float, cls: Int)] {
    let shape1 = feature.shape[1].intValue
    let numAnchors = feature.shape[2].intValue
    let numClasses = shape1 - 5

    let pointer = feature.dataPointer.bindMemory(to: Float.self, capacity: feature.count)
    let inputW = Float(modelInputSize.width)
    let inputH = Float(modelInputSize.height)

    struct Detection {
      let obb: OBB
      let score: Float
      let cls: Int
    }

    struct PointerWrapper: @unchecked Sendable {
      let pointer: UnsafeMutablePointer<Float>
    }

    struct DetectionsWrapper: @unchecked Sendable {
      let detections: UnsafeMutablePointer<Detection?>
    }

    let pointerWrapper = PointerWrapper(pointer: pointer)
    let detectionsPtr = UnsafeMutablePointer<Detection?>.allocate(capacity: numAnchors)
    detectionsPtr.initialize(repeating: nil, count: numAnchors)
    defer {
      detectionsPtr.deinitialize(count: numAnchors)
      detectionsPtr.deallocate()
    }
    let detectionsWrapper = DetectionsWrapper(detections: detectionsPtr)

    DispatchQueue.concurrentPerform(iterations: numAnchors) { i in
      let cx = pointerWrapper.pointer[i] / inputW
      let cy = pointerWrapper.pointer[numAnchors + i] / inputH
      let w = pointerWrapper.pointer[2 * numAnchors + i] / inputW
      let h = pointerWrapper.pointer[3 * numAnchors + i] / inputH

      var bestScore: Float = 0
      var bestClass: Int = 0
      for c in 0..<numClasses {
        let sc = pointerWrapper.pointer[(4 + c) * numAnchors + i]
        if sc > bestScore {
          bestScore = sc
          bestClass = c
        }
      }

      let angle = pointerWrapper.pointer[(4 + numClasses) * numAnchors + i]
      if bestScore > confidenceThreshold {
        detectionsWrapper.detections[i] = Detection(
          obb: OBB(cx: cx, cy: cy, w: w, h: h, angle: angle),
          score: bestScore, cls: bestClass)
      }
    }

    let detections = Array(UnsafeBufferPointer(start: detectionsPtr, count: numAnchors)).compactMap
    {
      $0
    }

    // OBB NMS
    let boxes = detections.map { $0.obb }
    let scores = detections.map { $0.score }
    let keep = nonMaxSuppressionOBB(boxes: boxes, scores: scores, iouThreshold: iouThreshold)

    return keep.map { idx in (detections[idx].obb, detections[idx].score, detections[idx].cls) }
  }

  private func nonMaxSuppressionOBB(
    boxes: [OBB], scores: [Float], iouThreshold: Float
  ) -> [Int] {
    let sortedIndices = scores.enumerated().sorted { $0.element > $1.element }.map { $0.offset }
    let precomputed: [OBBInfo] = boxes.map { OBBInfo($0) }
    var selected = [Int]()
    var active = [Bool](repeating: true, count: boxes.count)

    for i in 0..<sortedIndices.count {
      let idx = sortedIndices[i]
      if !active[idx] { continue }
      selected.append(idx)
      let boxA = precomputed[idx]
      for j in (i + 1)..<sortedIndices.count {
        let idxB = sortedIndices[j]
        if active[idxB] && boxA.aabbIntersects(with: precomputed[idxB]) {
          if boxA.iou(with: precomputed[idxB]) > iouThreshold {
            active[idxB] = false
          }
        }
      }
    }
    return selected
  }
}
