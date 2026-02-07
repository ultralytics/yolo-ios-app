// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import Foundation

/// Performs non-maximum suppression on bounding boxes to eliminate duplicate detections.
/// Used for legacy YOLO11 models that require NMS post-processing.
public func nonMaxSuppression(boxes: [CGRect], scores: [Float], threshold: Float) -> [Int] {
  let sortedIndices = scores.enumerated().sorted { $0.element > $1.element }.map { $0.offset }
  var selectedIndices = [Int]()
  var activeIndices = [Bool](repeating: true, count: boxes.count)

  for i in 0..<sortedIndices.count {
    let idx = sortedIndices[i]
    if activeIndices[idx] {
      selectedIndices.append(idx)
      for j in i + 1..<sortedIndices.count {
        let otherIdx = sortedIndices[j]
        if activeIndices[otherIdx] {
          let intersection = boxes[idx].intersection(boxes[otherIdx])
          if intersection.area > CGFloat(threshold) * min(boxes[idx].area, boxes[otherIdx].area) {
            activeIndices[otherIdx] = false
          }
        }
      }
    }
  }
  return selectedIndices
}

extension CGRect {
  var area: CGFloat {
    width * height
  }
}

// MARK: - OBB NMS

/// Polygon intersection using Sutherland-Hodgman algorithm.
func polygonIntersection(subjectPolygon: Polygon, clipPolygon: Polygon) -> Polygon {
  var outputList = subjectPolygon
  let clipEdgeCount = clipPolygon.count
  let closedClipPolygon = clipPolygon + [clipPolygon[0]]

  for i in 0..<clipEdgeCount {
    let currentEdgeStart = closedClipPolygon[i]
    let currentEdgeEnd = closedClipPolygon[i + 1]
    let inputList = outputList
    outputList = []
    if inputList.isEmpty { break }
    let closedInputPolygon = inputList + [inputList[0]]

    for j in 0..<(closedInputPolygon.count - 1) {
      let currentPoint = closedInputPolygon[j]
      let nextPoint = closedInputPolygon[j + 1]
      let currentInside = isInside(
        point: currentPoint, edgeStart: currentEdgeStart, edgeEnd: currentEdgeEnd)
      let nextInside = isInside(
        point: nextPoint, edgeStart: currentEdgeStart, edgeEnd: currentEdgeEnd)

      if currentInside && nextInside {
        outputList.append(nextPoint)
      } else if currentInside && !nextInside {
        if let intersec = computeIntersection(
          p1: currentPoint, p2: nextPoint, clipStart: currentEdgeStart, clipEnd: currentEdgeEnd)
        {
          outputList.append(intersec)
        }
      } else if !currentInside && nextInside {
        if let intersec = computeIntersection(
          p1: currentPoint, p2: nextPoint, clipStart: currentEdgeStart, clipEnd: currentEdgeEnd)
        {
          outputList.append(intersec)
        }
        outputList.append(nextPoint)
      }
    }
  }
  return outputList
}

private func isInside(point: CGPoint, edgeStart: CGPoint, edgeEnd: CGPoint) -> Bool {
  let cross =
    (edgeEnd.x - edgeStart.x) * (point.y - edgeStart.y)
    - (edgeEnd.y - edgeStart.y) * (point.x - edgeStart.x)
  return cross >= 0
}

private func computeIntersection(
  p1: CGPoint, p2: CGPoint, clipStart: CGPoint, clipEnd: CGPoint
) -> CGPoint? {
  let denom =
    (p1.x - p2.x) * (clipStart.y - clipEnd.y) - (p1.y - p2.y) * (clipStart.x - clipEnd.x)
  if abs(denom) < 1e-10 { return nil }
  let t =
    ((p1.x - clipStart.x) * (clipStart.y - clipEnd.y)
      - (p1.y - clipStart.y) * (clipStart.x - clipEnd.x)) / denom
  return CGPoint(x: p1.x + t * (p2.x - p1.x), y: p1.y + t * (p2.y - p1.y))
}

func polygonArea(_ poly: Polygon) -> CGFloat {
  guard poly.count > 2 else { return 0 }
  var area: CGFloat = 0
  for i in 0..<(poly.count - 1) {
    area += poly[i].x * poly[i + 1].y - poly[i + 1].x * poly[i].y
  }
  area += poly[poly.count - 1].x * poly[0].y - poly[0].x * poly[poly.count - 1].y
  return abs(area) * 0.5
}

/// Cached geometry for faster OBB IoU checks.
public struct OBBInfo: Sendable {
  let box: OBB
  let polygon: Polygon
  let area: CGFloat
  let aabb: CGRect

  public init(_ obb: OBB) {
    self.box = obb
    self.polygon = obb.toPolygon()
    self.area = CGFloat(obb.area)
    self.aabb = obb.toAABB()
  }

  func aabbIntersects(with other: OBBInfo) -> Bool {
    aabb.intersects(other.aabb)
  }

  func iou(with other: OBBInfo) -> Float {
    let interPoly = polygonIntersection(subjectPolygon: polygon, clipPolygon: other.polygon)
    let interArea = polygonArea(interPoly)
    let unionArea = area + other.area - interArea
    guard unionArea > 0 else { return 0 }
    return Float(interArea / unionArea)
  }
}
