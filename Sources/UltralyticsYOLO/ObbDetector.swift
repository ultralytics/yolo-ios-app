// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO SDK, implementing oriented bounding box (OBB) detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  ObbDetector detects objects at arbitrary orientations, predicting center, width, height, and rotation angle for
//  each box. Supports both traditional and YOLO26 end2end OBB output formats and includes a fast NMS that caches each
//  box's polygon, area, and axis-aligned bounding box, using AABB overlap as a cheap filter before computing polygon
//  intersection via the Sutherland–Hodgman algorithm.

import Accelerate
import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO models that detect objects using oriented (rotated) bounding boxes.
public final class ObbDetector: BasePredictor, @unchecked Sendable {

  override func processObservations(for request: VNRequest, _ error: Error?) {
    markInferenceEnd()
    guard let prediction = firstFeatureArray(request) else {
      self.isUpdating = false
      return
    }
    let obbResults = buildResults(from: prediction)
    self.updateTime()
    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], obb: obbResults,
      speed: self.t2, fps: 1 / self.t4, names: labels)
    applyTimingBreakdown(&result, smoothed: true)
    result.originalImage = currentOriginalImage
    self.currentOnResultsListener?.on(result: result)
  }

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }
    let requestHandler = makeRequestHandler(for: image)

    guard perform(request, with: requestHandler, errorMessage: "OBB detection failed"),
      let prediction = firstFeatureArray(request)
    else {
      return YOLOResult(
        orig_shape: inputSize, boxes: [], speed: finishTiming(notify: false), names: labels)
    }
    markInferenceEnd()
    let obbResults = buildResults(from: prediction)
    let speed = finishTiming(notify: false)  // before drawing: annotation is excluded from timings
    let annotatedImage = drawOBBsOnCIImage(ciImage: image, obbDetections: obbResults)
    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], obb: obbResults,
      annotatedImage: annotatedImage,
      speed: speed, names: labels)
    applyTimingBreakdown(&result)
    if capturesOriginalImage {
      result.originalImage = UIImage(ciImage: image)
    }
    return result
  }

  private func firstFeatureArray(_ request: VNRequest) -> MLMultiArray? {
    (request.results as? [VNCoreMLFeatureValueObservation])?.first?.featureValue.multiArrayValue
  }

  private func buildResults(from prediction: MLMultiArray) -> [OBBResult] {
    let nmsResults = postProcessOBB(
      feature: prediction,
      confidenceThreshold: Float(self.confidenceThreshold),
      iouThreshold: Float(self.iouThreshold))
    return nmsResults.prefix(self.numItemsThreshold).map { result in
      return OBBResult(
        box: inputOBB(fromModelOBB: result.box), confidence: result.score,
        cls: labelName(for: result.cls), index: result.cls)
    }
  }

  func postProcessOBB(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(box: OBB, score: Float, cls: Int)] {
    let shape = feature.shape.map { $0.intValue }
    guard shape.count == 3 else { return [] }

    // YOLO26 end2end OBB: [1, max_det, 7] where 7 = cx,cy,w,h,conf,class_id,angle (center xywh, NOT xyxy)
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
    let strides = feature.strides.map { $0.intValue }
    let channelStride = strides[1]
    let anchorStride = strides[2]
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
      let anchorOffset = i * anchorStride
      let cx = pointerWrapper.pointer[anchorOffset] / inputW
      let cy = pointerWrapper.pointer[channelStride + anchorOffset] / inputH
      let w = pointerWrapper.pointer[2 * channelStride + anchorOffset] / inputW
      let h = pointerWrapper.pointer[3 * channelStride + anchorOffset] / inputH

      // Find best class & score
      var bestScore: Float = 0
      var bestClass: Int = 0
      for c in 0..<numClasses {
        let sc = pointerWrapper.pointer[(4 + c) * channelStride + anchorOffset]
        if sc > bestScore {
          bestScore = sc
          bestClass = c
        }
      }

      // Angle is the last channel
      let angleIndex = (4 + numClasses) * channelStride + anchorOffset
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

  /// Processes YOLO26 end2end OBB output `[1, max_det, 7]` where each detection is
  /// `[cx, cy, w, h, conf, class_id, angle]` in pixel coords. OBB uses `dist2rbox()` which always outputs center-based
  /// xywh (not xyxy like detect). NMS is already applied by the model.
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

  /// Fast NMS for oriented bounding boxes. Caches each box's polygon, area, and axis-aligned bounding box, then uses a
  /// cheap AABB overlap check before computing polygon intersection IoU.
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

/// Returns the polygon formed by clipping subjectPolygon to clipPolygon using the Sutherland–Hodgman algorithm.
func polygonIntersection(subjectPolygon: Polygon, clipPolygon: Polygon) -> Polygon {
  var outputList = subjectPolygon
  let clipEdgeCount = clipPolygon.count

  // Close the polygon by appending the first point so we can iterate edges easily.
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

/// Returns true if `point` lies in the half-plane to the left of (or on) the directed edge `edgeStart` → `edgeEnd`.
///
/// Sutherland–Hodgman keeps the polygon on the left of each clip edge, assuming a clockwise clip polygon.
func isInside(
  point: CGPoint,
  edgeStart: CGPoint,
  edgeEnd: CGPoint
) -> Bool {
  // Sign of the 2D cross product (edgeEnd - edgeStart) x (point - edgeStart) indicates the side.
  let cross =
    (edgeEnd.x - edgeStart.x) * (point.y - edgeStart.y)
    - (edgeEnd.y - edgeStart.y) * (point.x - edgeStart.x)
  return cross >= 0
}

/// Returns the intersection point between the line segment `p1`→`p2` and the infinite line through
/// `clipStart`→`clipEnd`, or `nil` if the lines are parallel.
///
/// Sutherland–Hodgman uses this in a context that already classifies endpoints as inside/outside, so `t` is not
/// clamped to `[0, 1]`.
func computeIntersection(
  p1: CGPoint,
  p2: CGPoint,
  clipStart: CGPoint,
  clipEnd: CGPoint
) -> CGPoint? {
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
    // Lines are parallel (or nearly so)
    return nil
  }
  // Parametric intersection on p1→p2
  let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

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

/// Cached geometry (polygon, area, axis-aligned bounding box) for faster OBB IoU checks.
public struct OBBInfo {
  let polygon: Polygon  // The 4 corners in order
  let area: CGFloat
  let aabb: CGRect  // Axis-aligned bounding box for quick overlap check

  init(_ obb: OBB) {
    self.polygon = obb.toPolygon()
    self.area = CGFloat(obb.area)
    self.aabb = obb.toAABB()
  }

  /// Returns true if the axis-aligned bounding boxes overlap. If false, IoU is 0 and the expensive polygon
  /// intersection can be skipped.
  func aabbIntersects(with other: OBBInfo) -> Bool {
    return aabb.intersects(other.aabb)
  }

  /// Computes IoU between two OBBs using their polygon intersection.
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
  /// Returns the axis-aligned bounding box enclosing this rotated box.
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
