// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Foundation
import Vision

/// Specialized predictor for YOLO classification models.
public class Classifier: BasePredictor {

  override func processResults() -> YOLOResult {
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
    }

    var probs = Probs(top1: "", top5: [], top1Conf: 0, top5Confs: [])

    if let observation = request.results as? [VNCoreMLFeatureValueObservation] {
      if let multiArray = observation.first?.featureValue.multiArrayValue {
        var valuesArray = [Double]()
        for i in 0..<multiArray.count {
          valuesArray.append(multiArray[i].doubleValue)
        }

        var indexedMap = [Int: Double]()
        for (index, value) in valuesArray.enumerated() {
          indexedMap[index] = value
        }
        let sortedMap = indexedMap.sorted { $0.value > $1.value }

        if let (topIndex, topScore) = sortedMap.first {
          probs.top1 = labels[topIndex]
          probs.top1Conf = Float(topScore)
        }

        let topObservations = sortedMap.prefix(5)
        for (index, value) in topObservations {
          probs.top5.append(labels[index])
          probs.top5Confs.append(Float(value))
        }
      }
    } else if let observations = request.results as? [VNClassificationObservation] {
      let candidateNumber = min(5, observations.count)
      if let topObservation = observations.first {
        probs.top1 = topObservation.identifier
        probs.top1Conf = Float(topObservation.confidence)
      }
      for i in 0..<candidateNumber {
        probs.top5.append(observations[i].identifier)
        probs.top5Confs.append(Float(observations[i].confidence))
      }
    }

    return YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: t1,
      fps: t4 > 0 ? 1 / t4 : nil, names: labels)
  }
}
