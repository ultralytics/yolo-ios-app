//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  PostProcessing for Ultralytics YOLO App
// This feature is designed to post-process the output of a YOLOv8 model within the Ultralytics YOLO app to extract high-confidence objects.
// Output high confidence boxes and their corresponding feature values using Non max suppression.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import CoreML
import Foundation
import Vision

func nonMaxSuppression(boxes: [CGRect], scores: [Float], threshold: Float) -> [Int] {
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

// Human model's output [1,15,8400] to [(Box, Confidence, HumanFeatures)]

func PostProcessHuman(prediction: MLMultiArray, confidenceThreshold: Float, iouThreshold: Float)
  -> [(CGRect, Float, [Float])]
{
  let numAnchors = prediction.shape[2].intValue
  var boxes = [CGRect]()
  var scores = [Float]()
  var features = [[Float]]()
  let featurePointer = UnsafeMutablePointer<Float>(OpaquePointer(prediction.dataPointer))
  let lock = DispatchQueue(label: "com.example.lock")

  DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
    let confIndex = 4 * numAnchors + j
    let confidence = featurePointer[confIndex]
    if confidence > confidenceThreshold {
      let x = featurePointer[j]
      let y = featurePointer[numAnchors + j]
      let width = featurePointer[2 * numAnchors + j]
      let height = featurePointer[3 * numAnchors + j]

      let boxWidth = CGFloat(width)
      let boxHeight = CGFloat(height)
      let boxX = CGFloat(x - width / 2)
      let boxY = CGFloat(y - height / 2)

      let boundingBox = CGRect(x: boxX, y: boxY, width: boxWidth, height: boxHeight)

      var boxFeatures = [Float](repeating: 0, count: 11)
      for k in 0..<11 {
        let key = (5 + k) * numAnchors + j
        boxFeatures[k] = featurePointer[key]
      }

      lock.sync {
        boxes.append(boundingBox)
        scores.append(confidence)
        features.append(boxFeatures)
      }
    }
  }

  let selectedIndices = nonMaxSuppression(boxes: boxes, scores: scores, threshold: iouThreshold)
  var selectedBoxesAndFeatures = [(CGRect, Float, [Float])]()

  for idx in selectedIndices {
    selectedBoxesAndFeatures.append((boxes[idx], scores[idx], features[idx]))
  }
  print(selectedBoxesAndFeatures)
  return selectedBoxesAndFeatures
}

func toPerson(boxesAndScoresAndFeatures: [(CGRect, Float, [Float])]) -> [Person] {
  var persons = [Person]()
  for detectedHuman in boxesAndScoresAndFeatures {
    var person = Person(index: -1)
    person.update(box: detectedHuman.0, score: detectedHuman.1, features: detectedHuman.2)
    person.color = .red
    persons.append(person)
  }
  return persons
}

extension CGRect {
  var area: CGFloat {
    return width * height
  }
}
