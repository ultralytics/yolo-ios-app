// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, implementing oriented bounding box detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ObbDetector class provides functionality for detecting objects with oriented (rotated)
//  bounding boxes. Unlike standard object detection that uses axis-aligned boxes, this implementation
//  handles objects at arbitrary orientations by predicting the center, width, height, and rotation
//  angle of each bounding box. The class includes specialized algorithms for non-maximum suppression
//  of oriented boxes, computing polygon intersections using the Sutherland-Hodgman algorithm, and
//  efficient IoU calculations. These optimizations enable real-time performance even when dealing
//  with the computational complexity of rotated geometry operations.

import Accelerate
import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO models that detect objects using oriented (rotated) bounding boxes.
public class ObbDetector: BasePredictor, @unchecked Sendable {

  override func processObservations(for request: VNRequest, error: Error?) {
    if let results = request.results as? [VNCoreMLFeatureValueObservation] {

      if let prediction = results.first?.featureValue.multiArrayValue {
        let nmsResults = postProcessOBB(
          feature: prediction,  // your MLMultiArray
          confidenceThreshold: Float(self.confidenceThreshold),
          iouThreshold: Float(self.iouThreshold)
        )

        var obbResults: [OBBResult] = []
        let limitedResults = nmsResults.prefix(self.numItemsThreshold)
        for result in limitedResults {
          let box = result.box
          let score = result.score
          guard result.cls < labels.count else { continue }
          let clsIdx = labels[result.cls]
          let obbResult = OBBResult(box: box, confidence: score, cls: clsIdx, index: result.cls)
          obbResults.append(obbResult)
        }

        self.updateTime()
        self.currentOnResultsListener?.on(
          result: YOLOResult(
            orig_shape: inputSize, boxes: [], obb: obbResults, speed: self.t2, fps: 1 / self.t4,
            names: labels))
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

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
      return emptyResult
    }
    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    let result = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)

    do {
      try requestHandler.perform([request])

      if let results = request.results as? [VNCoreMLFeatureValueObservation] {

        if let prediction = results.first?.featureValue.multiArrayValue {
          let nmsResults = postProcessOBB(
            feature: prediction,  // your MLMultiArray
            confidenceThreshold: Float(self.confidenceThreshold),
            iouThreshold: Float(self.iouThreshold)
          )

          var obbResults: [OBBResult] = []
          let limitedResults = nmsResults.prefix(self.numItemsThreshold)
          for result in limitedResults {
            let box = result.box
            let score = result.score
            guard result.cls < labels.count else { continue }
            let clsIdx = labels[result.cls]
            let obbResult = OBBResult(box: box, confidence: score, cls: clsIdx, index: result.cls)
            obbResults.append(obbResult)
          }
          let annotatedImage = drawOBBsOnCIImage(ciImage: image, obbDetections: obbResults)
          updateTime()
          return YOLOResult(
            orig_shape: inputSize, boxes: [], masks: nil, probs: nil, keypointsList: [],
            obb: obbResults, annotatedImage: annotatedImage, speed: self.t2, fps: 1 / self.t4,
            names: labels)
        }
      }
    } catch {
      print(error)
    }
    return result
  }

  func postProcessOBB(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(box: OBB, score: Float, cls: Int)] {
    let shape = feature.shape.map { $0.intValue }
    guard shape.count == 3 else { return [] }

    // YOLO26 end2end OBB: [1, max_det, 7] where 7 = x1,y1,x2,y2,conf,class_id,angle
    // Traditional OBB: [1, 4+nc+1, num_anchors] where shape[2] > shape[1]
    if shape[2] < shape[1] {
      return postProcessEnd2EndOBB(
        feature: feature, shape: shape, confidenceThreshold: confidenceThreshold)
    }

    let shape1 = shape[1]
    let numAnchors = shape[2]
    let numClasses = shape1 - 5  // (4 + numClasses + 1) = shape1

    let pointer = feature.dataPointer.bindMemory(
      to: Float.self,
      capacity: feature.count)
    let inputW = Float(modelInputSize.width)
    let inputH = Float(modelInputSize.height)

    struct Detection {
      let obb: OBB
      let score: Float
      let cls: Int
    }

    // Wrapper to make pointer and array Sendable for Swift 6
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

    // 1) Parallel-extract predictions
    DispatchQueue.concurrentPerform(iterations: numAnchors) { i in
      let cx = pointerWrapper.pointer[i] / inputW
      let cy = pointerWrapper.pointer[numAnchors + i] / inputH
      let w = pointerWrapper.pointer[2 * numAnchors + i] / inputW
      let h = pointerWrapper.pointer[3 * numAnchors + i] / inputH

      // Find best class & score
      var bestScore: Float = 0
      var bestClass: Int = 0
      for c in 0..<numClasses {
        let sc = pointerWrapper.pointer[(4 + c) * numAnchors + i]
        if sc > bestScore {
          bestScore = sc
          bestClass = c
        }
      }

      // Angle is the last channel
      let angleIndex = (4 + numClasses) * numAnchors + i
      let angle = pointerWrapper.pointer[angleIndex]

      // Threshold
      if bestScore > confidenceThreshold {
        let obb = OBB(cx: cx, cy: cy, w: w, h: h, angle: angle)
        detectionsWrapper.detections[i] = Detection(obb: obb, score: bestScore, cls: bestClass)
      }
    }

    // Convert pointer array to Swift array
    let rawDetections = Array(UnsafeBufferPointer(start: detectionsPtr, count: numAnchors))
    let detections = rawDetections.compactMap { $0 }  // remove nil

    // 2) Run faster OBB NMS
    let boxes = detections.map { $0.obb }
    let scores = detections.map { $0.score }
    let keep = nonMaxSuppressionOBB(
      boxes: boxes,
      scores: scores,
      iouThreshold: iouThreshold)

    // 3) Build final
    var results = [(box: OBB, score: Float, cls: Int)]()
    results.reserveCapacity(keep.count)
    for idx in keep {
      let d = detections[idx]
      results.append((d.obb, d.score, d.cls))
    }
    return results
  }

  /// Processes YOLO26 end2end OBB output: [1, max_det, 7].
  /// Each detection: [cx, cy, w, h, conf, class_id, angle] in xywh pixel coords.
  /// OBB uses dist2rbox() which always outputs center-based xywh (NOT xyxy like detect).
  /// NMS is already applied by the model.
  private func postProcessEnd2EndOBB(
    feature: MLMultiArray,
    shape: [Int],
    confidenceThreshold: Float
  ) -> [(box: OBB, score: Float, cls: Int)] {
    let numDetections = shape[1]
    let numFields = shape[2]
    let strides = feature.strides.map { $0.intValue }
    let pointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let detStride = strides[1]
    let fieldStride = strides[2]
    let inputW = Float(modelInputSize.width)
    let inputH = Float(modelInputSize.height)

    var results: [(box: OBB, score: Float, cls: Int)] = []

    for i in 0..<numDetections {
      let base = i * detStride
      let conf = pointer[base + 4 * fieldStride]
      guard conf > confidenceThreshold else { continue }

      // OBB boxes are center-based xywh from dist2rbox (NOT xyxy like detect)
      let cx = pointer[base] / inputW
      let cy = pointer[base + fieldStride] / inputH
      let w = pointer[base + 2 * fieldStride] / inputW
      let h = pointer[base + 3 * fieldStride] / inputH
      let classId = numFields > 6 ? Int(pointer[base + 5 * fieldStride]) : 0

      // Angle is the last field — OBB26 outputs raw angle in radians (no sigmoid needed,
      // model learns to predict radians directly, used with cos/sin in dist2rbox)
      let angle = pointer[base + (numFields - 1) * fieldStride]

      let obb = OBB(cx: cx, cy: cy, w: w, h: h, angle: angle)
      results.append((box: obb, score: conf, cls: classId))
    }

    return results
  }

  /// Fast NMS for oriented bounding boxes.
  /// Internally, it caches each box's polygon, area, and axis-aligned bounding box.
  /// Then it does a quick AABB overlap check before polygon intersection to skip expensive IoU.
  public func nonMaxSuppressionOBB(
    boxes: [OBB],
    scores: [Float],
    iouThreshold: Float
  ) -> [Int] {
    // 1) Sort boxes by descending confidence
    let sortedIndices = scores.enumerated()
      .sorted { $0.element > $1.element }
      .map { $0.offset }

    // 2) Precompute geometry for each OBB to speed up IoU checks
    let precomputed: [OBBInfo] = boxes.map { OBBInfo($0) }

    var selected: [Int] = []
    selected.reserveCapacity(boxes.count)

    var active = [Bool](repeating: true, count: boxes.count)

    // 3) NMS
    for i in 0..<sortedIndices.count {
      let idx = sortedIndices[i]
      if !active[idx] {
        continue
      }
      // This box survives => keep it
      selected.append(idx)

      // Compare with lower-score boxes
      let boxA = precomputed[idx]
      for j in (i + 1)..<sortedIndices.count {
        let idxB = sortedIndices[j]
        if active[idxB] {
          // Cheap bounding-box check first
          if boxA.aabbIntersects(with: precomputed[idxB]) {
            // Then do real IoU
            let iouVal = boxA.iou(with: precomputed[idxB])
            if iouVal > iouThreshold {
              active[idxB] = false
            }
          }
        }
      }
    }

    return selected
  }

}

/// Returns the polygon formed by clipping subjectPolygon to clipPolygon using
/// the Sutherland–Hodgman algorithm.
func polygonIntersection(subjectPolygon: Polygon, clipPolygon: Polygon) -> Polygon {
  var outputList = subjectPolygon
  let clipEdgeCount = clipPolygon.count

  // If the polygon is not closed, append the first point at the end
  // so we can iterate edges easily.
  let closedClipPolygon = clipPolygon + [clipPolygon[0]]

  for i in 0..<clipEdgeCount {
    let currentEdgeStart = closedClipPolygon[i]
    let currentEdgeEnd = closedClipPolygon[i + 1]

    let inputList = outputList
    outputList = []

    if inputList.isEmpty { break }

    // Also close the input polygon for easy iteration
    let closedInputPolygon = inputList + [inputList[0]]

    for j in 0..<(closedInputPolygon.count - 1) {
      let currentPoint = closedInputPolygon[j]
      let nextPoint = closedInputPolygon[j + 1]

      let currentInside = isInside(
        point: currentPoint,
        edgeStart: currentEdgeStart,
        edgeEnd: currentEdgeEnd)
      let nextInside = isInside(
        point: nextPoint,
        edgeStart: currentEdgeStart,
        edgeEnd: currentEdgeEnd)

      if currentInside && nextInside {
        // Both inside
        outputList.append(nextPoint)
      } else if currentInside && !nextInside {
        // Going outside -> compute intersection
        if let intersec = computeIntersection(
          p1: currentPoint,
          p2: nextPoint,
          clipStart: currentEdgeStart,
          clipEnd: currentEdgeEnd)
        {
          outputList.append(intersec)
        }
      } else if !currentInside && nextInside {
        // Going inside -> add intersection + next
        if let intersec = computeIntersection(
          p1: currentPoint,
          p2: nextPoint,
          clipStart: currentEdgeStart,
          clipEnd: currentEdgeEnd)
        {
          outputList.append(intersec)
        }
        outputList.append(nextPoint)
      }
      // else both outside -> add nothing
    }
  }

  return outputList
}

/// Determine if a point is inside the "half-plane" defined by the directed edge
/// from edgeStart -> edgeEnd. We treat "left-turn" or "right-turn" checks.
func isInside(
  point: CGPoint,
  edgeStart: CGPoint,
  edgeEnd: CGPoint
) -> Bool {
  // We can use cross product to check which side of the directed edge we are on.
  // For a standard "clip left" convention, use the sign of the cross product:
  // Vector(edgeEnd - edgeStart) x Vector(point - edgeStart).
  let cross =
    (edgeEnd.x - edgeStart.x) * (point.y - edgeStart.y)
    - (edgeEnd.y - edgeStart.y) * (point.x - edgeStart.x)
  // If cross >= 0, point is to the left or on the line.
  // This depends on which side we treat as "inside".
  // Sutherland–Hodgman typically keeps polygon inside to the left of each edge
  // (assuming a clockwise clip polygon).
  return cross >= 0
}

/// Compute intersection between the line segment p1->p2 and the infinite line
/// defined by clipStart->clipEnd.
func computeIntersection(
  p1: CGPoint,
  p2: CGPoint,
  clipStart: CGPoint,
  clipEnd: CGPoint
) -> CGPoint? {
  // Solve for parametric intersection.
  // Parametric line eqn:
  //   p1 + t (p2 - p1)
  //   clipStart + u (clipEnd - clipStart)
  // We want to find t where lines intersect in 2D. Then check if 0..1 for the segment.

  let x1 = p1.x
  let y1 = p1.y
  let x2 = p2.x
  let y2 = p2.y
  let x3 = clipStart.x
  let y3 = clipStart.y
  let x4 = clipEnd.x
  let y4 = clipEnd.y

  let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
  if abs(denom) < 1e-10 {
    // Lines are parallel or extremely close to parallel
    return nil
  }
  // Intersection parameter t on p1->p2
  let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

  // We only need the point, not strictly to check 0<=t<=1,
  // because Sutherland–Hodgman calls this in a context where
  // it decides how to treat inside vs outside.
  // But it's nice to be consistent:
  // if t < 0 or t > 1 => intersection not on the segment p1->p2.
  // For standard polygon clipping, we still use the intersection anyway.
  // So we won't clamp T here, but you can if you want to discard outside segments.

  let ix = x1 + t * (x2 - x1)
  let iy = y1 + t * (y2 - y1)

  return CGPoint(x: ix, y: iy)
}

func polygonArea(_ poly: Polygon) -> CGFloat {
  guard poly.count > 2 else { return 0 }
  var area: CGFloat = 0
  for i in 0..<(poly.count - 1) {
    area += poly[i].x * poly[i + 1].y - poly[i + 1].x * poly[i].y
  }
  // Close the polygon
  area += poly[poly.count - 1].x * poly[0].y - poly[0].x * poly[poly.count - 1].y
  return abs(area) * 0.5
}

/// Store cached geometry for faster OBB IoU checks.
public struct OBBInfo {
  let box: OBB
  let polygon: Polygon  // The 4 corners in order
  let area: CGFloat
  let aabb: CGRect  // Axis-aligned bounding box for quick overlap check

  init(_ obb: OBB) {
    self.box = obb
    self.polygon = obb.toPolygon()
    self.area = CGFloat(obb.area)
    self.aabb = obb.toAABB()
  }

  /// Quickly check if the axis-aligned bounding boxes intersect.
  /// If not, the IoU is 0, so we can skip the expensive polygon intersection.
  func aabbIntersects(with other: OBBInfo) -> Bool {
    return aabb.intersects(other.aabb)
  }

  /// Compute actual IoU using polygon intersection if bounding boxes overlap.
  func iou(with other: OBBInfo) -> Float {
    let interPoly = polygonIntersection(
      subjectPolygon: polygon,
      clipPolygon: other.polygon)
    let interArea = polygonArea(interPoly)
    let unionArea = area + other.area - interArea
    guard unionArea > 0 else { return 0 }
    return Float(interArea / unionArea)
  }
}

extension OBB {
  /// Return the axis-aligned bounding box around this rotated box
  func toAABB() -> CGRect {
    let poly = toPolygon()
    var minX = CGFloat.infinity
    var maxX = -CGFloat.infinity
    var minY = CGFloat.infinity
    var maxY = -CGFloat.infinity
    for p in poly {
      if p.x < minX { minX = p.x }
      if p.x > maxX { maxX = p.x }
      if p.y < minY { minY = p.y }
      if p.y > maxY { maxY = p.y }
    }
    return CGRect(
      x: minX, y: minY,
      width: maxX - minX,
      height: maxY - minY)
  }
}
