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

import Accelerate
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO classification models that identify the subject of an image.
public class Classifier: BasePredictor, @unchecked Sendable {

  override func processObservations(for request: VNRequest, error: Error?) {
    let imageWidth = inputSize.width
    let imageHeight = inputSize.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    var probs = Probs(top1: "", top5: [], top1Conf: 0, top5Confs: [])

    if let observation = request.results as? [VNCoreMLFeatureValueObservation] {

      // Get the MLMultiArray from the observation
      let multiArray = observation.first?.featureValue.multiArrayValue

      if let multiArray = multiArray {
        // Apply softmax to convert raw logits to probabilities
        var floatValues = [Float](repeating: 0, count: multiArray.count)
        for i in 0..<multiArray.count {
          floatValues[i] = multiArray[i].floatValue
        }
        var softmaxOutput = [Float](repeating: 0, count: floatValues.count)
        var count = Int32(floatValues.count)
        vvexpf(&softmaxOutput, floatValues, &count)
        var sumExp: Float = 0
        vDSP_sve(softmaxOutput, 1, &sumExp, vDSP_Length(floatValues.count))
        if sumExp > 0 {
          vDSP_vsdiv(softmaxOutput, 1, &sumExp, &softmaxOutput, 1, vDSP_Length(floatValues.count))
        }

        var indexedMap = [Int: Double]()
        for (index, value) in softmaxOutput.enumerated() {
          indexedMap[index] = Double(value)
        }

        let sortedMap = indexedMap.sorted { $0.value > $1.value }

        // top1
        if let (topIndex, topScore) = sortedMap.first, topIndex < labels.count {
          let top1Label = labels[topIndex]
          let top1Conf = Float(topScore)
          probs.top1 = top1Label
          probs.top1Conf = top1Conf
        }

        // top5
        let topObservations = sortedMap.prefix(5)
        var top5Labels: [String] = []
        var top5Confs: [Float] = []

        for (index, value) in topObservations where index < labels.count {
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
      for i in 0..<candidateNumber {
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
          // Apply softmax to convert raw logits to probabilities
          var floatValues = [Float](repeating: 0, count: multiArray.count)
          for i in 0..<multiArray.count {
            floatValues[i] = multiArray[i].floatValue
          }
          var softmaxOutput = [Float](repeating: 0, count: floatValues.count)
          var count = Int32(floatValues.count)
          vvexpf(&softmaxOutput, floatValues, &count)
          var sumExp: Float = 0
          vDSP_sve(softmaxOutput, 1, &sumExp, vDSP_Length(floatValues.count))
          if sumExp > 0 {
            vDSP_vsdiv(softmaxOutput, 1, &sumExp, &softmaxOutput, 1, vDSP_Length(floatValues.count))
          }

          var indexedMap = [Int: Double]()
          for (index, value) in softmaxOutput.enumerated() {
            indexedMap[index] = Double(value)
          }

          let sortedMap = indexedMap.sorted { $0.value > $1.value }

          // top1
          if let (topIndex, topScore) = sortedMap.first, topIndex < labels.count {
            let top1Label = labels[topIndex]
            let top1Conf = Float(topScore)
            probs.top1 = top1Label
            probs.top1Conf = top1Conf
          }

          // top5
          let topObservations = sortedMap.prefix(5)
          var top5Labels: [String] = []
          var top5Confs: [Float] = []

          for (index, value) in topObservations where index < labels.count {
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
        for i in 0..<candidateNumber {
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
