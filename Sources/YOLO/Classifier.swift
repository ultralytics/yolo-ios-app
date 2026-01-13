// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, implementing image classification functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Classifier class implements image classification using YOLO models. Unlike object detection
//  or segmentation, it focuses on identifying the primary subject of an image rather than locating
//  objects within it. The class processes model outputs to extract classification probabilities,
//  identifying the top predicted class and confidence score. It supports multiple output formats
//  from Vision framework requests, handling both VNCoreMLFeatureValueObservation and
//  VNClassificationObservation result types. The implementation extracts both the top prediction
//  and the top 5 predictions with their confidence scores, enabling rich user feedback.

import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO classification models that identify the subject of an image.
public class Classifier: BasePredictor, @unchecked Sendable {

  /// Checks if the current model is a YOLO26 model
  private var isYOLO26Model: Bool {
    guard let url = modelURL else { return false }
    let fullPath = url.path.lowercased()
    let modelName = url.lastPathComponent.lowercased()
    let baseName =
      modelName
      .replacingOccurrences(of: ".mlmodelc", with: "")
      .replacingOccurrences(of: ".mlpackage", with: "")
      .replacingOccurrences(of: ".mlmodel", with: "")
    return fullPath.contains("yolo26") || baseName.contains("yolo26")
  }

  override func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  override func setIouThreshold(iou: Double) {
    iouThreshold = iou
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  override func processObservations(for request: VNRequest, error: Error?) {
    let imageWidth = inputSize.width
    let imageHeight = inputSize.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    var probs = Probs(top1: "", top5: [], top1Conf: 0, top5Confs: [])

    if let observation = request.results as? [VNCoreMLFeatureValueObservation] {

      // Get the MLMultiArray from the observation
      let multiArray = observation.first?.featureValue.multiArrayValue

      if let multiArray = multiArray {
        // Initialize an array to store the classes
        var valuesArray = [Double]()

        // Helper function to normalize confidence scores for YOLO26
        func normalizeConfidence(_ value: Double) -> Double {
          if isYOLO26Model {
            // YOLO26 might output in different formats:
            // 1. Logits (very large positive/negative) - apply sigmoid
            // 2. 0-100 range - normalize to 0-1
            // 3. Already 0-1 - use as-is
            if abs(value) > 10.0 {
              // Likely logits, apply sigmoid
              return 1.0 / (1.0 + exp(-value))
            } else if value > 1.0 && value <= 100.0 {
              // Likely 0-100 range, normalize to 0-1
              return value / 100.0
            }
          }
          // If already in 0-1 range or not YOLO26, use as-is
          return value
        }

        for i in 0..<multiArray.count {
          let rawValue = multiArray[i].doubleValue
          let normalizedValue = normalizeConfidence(rawValue)
          valuesArray.append(normalizedValue)
        }

        var indexedMap = [Int: Double]()
        for (index, value) in valuesArray.enumerated() {
          indexedMap[index] = value
        }

        let sortedMap = indexedMap.sorted { $0.value > $1.value }

        // top1
        if let (topIndex, topScore) = sortedMap.first {
          let top1Label = labels[topIndex]
          let top1Conf = Float(topScore)
          probs.top1 = top1Label
          probs.top1Conf = top1Conf
        }

        // top5
        let topObservations = sortedMap.prefix(5)
        var top5Labels: [String] = []
        var top5Confs: [Float] = []

        for (index, value) in topObservations {
          top5Labels.append(labels[index])
          top5Confs.append(Float(value))
        }

        probs.top5 = top5Labels
        probs.top5Confs = top5Confs
      }
    } else if let observations = request.results as? [VNClassificationObservation] {
      var top1 = ""
      var top1Conf: Float = 0
      var top5: [String] = []
      var top5Confs: [Float] = []

      var candidateNumber = 5
      if observations.count < candidateNumber {
        candidateNumber = observations.count
      }
      if let topObservation = observations.first {
        top1 = topObservation.identifier
        top1Conf = Float(topObservation.confidence)
      }
      for i in 0...candidateNumber - 1 {
        let observation = observations[i]
        let label = observation.identifier
        let confidence: Float = Float(observation.confidence)
        top5Confs.append(confidence)
        top5.append(label)
      }
      probs = Probs(top1: top1, top5: top5, top1Conf: top1Conf, top5Confs: top5Confs)
    }

    // Measure FPS
    if self.t1 < 10.0 {  // valid dt
      self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
    }
    self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
    self.t3 = CACurrentMediaTime()

    self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)  // t2 seconds to ms
    //                self.currentOnFpsRateListener?.on(fpsRate: 1 / self.t4)
    let result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: self.t2, fps: 1 / self.t4,
      names: labels)

    self.currentOnResultsListener?.on(result: result)

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
    var probs = Probs(top1: "", top5: [], top1Conf: 0, top5Confs: [])
    do {
      try requestHandler.perform([request])
      if let observation = request.results as? [VNCoreMLFeatureValueObservation] {
        // Get the MLMultiArray from the observation
        let multiArray = observation.first?.featureValue.multiArrayValue

        if let multiArray = multiArray {
          // Initialize an array to store the classes
          var valuesArray = [Double]()

          // Helper function to normalize confidence scores for YOLO26
          func normalizeConfidence(_ value: Double) -> Double {
            if isYOLO26Model {
              // YOLO26 might output in different formats:
              // 1. Logits (very large positive/negative) - apply sigmoid
              // 2. 0-100 range - normalize to 0-1
              // 3. Already 0-1 - use as-is
              if abs(value) > 10.0 {
                // Likely logits, apply sigmoid
                return 1.0 / (1.0 + exp(-value))
              } else if value > 1.0 && value <= 100.0 {
                // Likely 0-100 range, normalize to 0-1
                return value / 100.0
              }
            }
            // If already in 0-1 range or not YOLO26, use as-is
            return value
          }

          for i in 0..<multiArray.count {
            let rawValue = multiArray[i].doubleValue
            let normalizedValue = normalizeConfidence(rawValue)
            valuesArray.append(normalizedValue)
          }

          var indexedMap = [Int: Double]()
          for (index, value) in valuesArray.enumerated() {
            indexedMap[index] = value
          }

          let sortedMap = indexedMap.sorted { $0.value > $1.value }

          // top1
          if let (topIndex, topScore) = sortedMap.first {
            let top1Label = labels[topIndex]
            let top1Conf = Float(topScore)
            probs.top1 = top1Label
            probs.top1Conf = top1Conf
          }

          // top5
          let topObservations = sortedMap.prefix(5)
          var top5Labels: [String] = []
          var top5Confs: [Float] = []

          for (index, value) in topObservations {
            top5Labels.append(labels[index])
            top5Confs.append(Float(value))
          }

          probs.top5 = top5Labels
          probs.top5Confs = top5Confs
        }
      } else if let observations = request.results as? [VNClassificationObservation] {
        var top1 = ""
        var top1Conf: Float = 0
        var top5: [String] = []
        var top5Confs: [Float] = []

        var candidateNumber = 5
        if observations.count < candidateNumber {
          candidateNumber = observations.count
        }
        if let topObservation = observations.first {
          top1 = topObservation.identifier
          top1Conf = Float(topObservation.confidence)
        }
        for i in 0...candidateNumber - 1 {
          let observation = observations[i]
          let label = observation.identifier
          let confidence: Float = Float(observation.confidence)
          top5Confs.append(confidence)
          top5.append(label)
        }
        probs = Probs(top1: top1, top5: top5, top1Conf: top1Conf, top5Confs: top5Confs)
      }

    } catch {
      print(error)
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: t1, names: labels)
    let annotatedImage = drawYOLOClassifications(on: image, result: result)
    result.annotatedImage = annotatedImage
    return result
  }
}
