// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO SDK, providing utilities for post-processing detections.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  Filters redundant axis-aligned detections via non-maximum suppression. Candidates are sorted by confidence; the
//  highest-scoring box is kept and any lower-scoring box whose IoU (intersection-over-union) with it exceeds the
//  threshold is suppressed. Used after traditional (non-NMS-free) YOLO model decoding.

import Foundation

/// Performs non-maximum suppression on a set of bounding boxes to eliminate duplicate detections.
/// - Parameters:
///   - boxes: Array of bounding boxes in CGRect format.
///   - scores: Confidence scores corresponding to each bounding box.
///   - threshold: The minimum overlap ratio required to suppress a box.
/// - Returns: Indices of the selected bounding boxes after suppression.
public func nonMaxSuppression(boxes: [CGRect], scores: [Float], threshold: Float) -> [Int] {
  let count = boxes.count
  let sortedIndices = scores.enumerated().sorted { $0.element > $1.element }.map { $0.offset }
  let areas = boxes.map { $0.area }
  var selectedIndices = [Int]()
  var activeIndices = [Bool](repeating: true, count: count)
  let iouThreshold = CGFloat(threshold)

  for i in 0..<sortedIndices.count {
    let idx = sortedIndices[i]
    if !activeIndices[idx] { continue }
    selectedIndices.append(idx)
    let boxA = boxes[idx]
    let areaA = areas[idx]
    for j in (i + 1)..<sortedIndices.count {
      let otherIdx = sortedIndices[j]
      if !activeIndices[otherIdx] { continue }
      let interArea = boxA.intersection(boxes[otherIdx]).area
      let union = areaA + areas[otherIdx] - interArea
      if union > 0 && interArea / union > iouThreshold {
        activeIndices[otherIdx] = false
      }
    }
  }
  return selectedIndices
}

extension CGRect {
  var area: CGFloat {
    return width * height
  }
}
