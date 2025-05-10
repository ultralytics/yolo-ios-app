// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, providing utilities for post-processing detections.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The NonMaxSuppression utility provides a fundamental algorithm for filtering redundant
//  object detections. When multiple bounding boxes detect the same object, non-maximum suppression
//  selects the best detection and suppresses overlapping boxes with lower confidence scores.
//  This implementation sorts detection candidates by confidence score, then iteratively selects
//  the highest-scoring boxes while removing others that have significant overlap (measured by
//  intersection over minimum area). The algorithm is essential for producing clean detection
//  results by removing duplicate predictions.

import Foundation

/// Performs non-maximum suppression on a set of bounding boxes to eliminate duplicate detections.
/// - Parameters:
///   - boxes: Array of bounding boxes in CGRect format.
///   - scores: Confidence scores corresponding to each bounding box.
///   - threshold: The minimum overlap ratio required to suppress a box.
/// - Returns: Indices of the selected bounding boxes after suppression.
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
    return width * height
  }
}
