import Foundation

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
