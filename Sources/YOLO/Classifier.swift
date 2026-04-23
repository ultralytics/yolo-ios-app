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
public final class Classifier: BasePredictor, @unchecked Sendable {

  override func processObservations(for request: VNRequest, _ error: Error?) {
    let probs = extractProbs(from: request)
    self.updateTime()
    let result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: self.t2, fps: 1 / self.t4,
      names: labels)
    self.currentOnResultsListener?.on(result: result)
  }

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    self.inputSize = CGSize(width: image.extent.width, height: image.extent.height)
    var probs = Probs(top1: "", top5: [], top1Conf: 0, top5Confs: [])
    do {
      try requestHandler.perform([request])
      probs = extractProbs(from: request)
    } catch {
      YOLOLog.error("Classifier inference failed: \(error)")
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: t1, names: labels)
    result.annotatedImage = drawYOLOClassifications(on: image, result: result)
    return result
  }

  /// Extracts top-1 and top-5 probabilities from a Vision request result.
  ///
  /// Handles both `VNCoreMLFeatureValueObservation` (raw logits that require softmax) and
  /// `VNClassificationObservation` (already-normalized scores).
  private func extractProbs(from request: VNRequest) -> Probs {
    if let observations = request.results as? [VNCoreMLFeatureValueObservation],
      let multiArray = observations.first?.featureValue.multiArrayValue
    {
      return softmaxProbs(from: multiArray)
    }
    if let observations = request.results as? [VNClassificationObservation] {
      let top = observations.prefix(5)
      return Probs(
        top1: observations.first?.identifier ?? "",
        top5: top.map { $0.identifier },
        top1Conf: Float(observations.first?.confidence ?? 0),
        top5Confs: top.map { Float($0.confidence) }
      )
    }
    return Probs(top1: "", top5: [], top1Conf: 0, top5Confs: [])
  }

  /// Applies softmax to raw logits and returns the top-1/top-5 probabilities.
  private func softmaxProbs(from multiArray: MLMultiArray) -> Probs {
    let count = multiArray.count
    var logits = [Float](repeating: 0, count: count)
    for i in 0..<count { logits[i] = multiArray[i].floatValue }

    var output = [Float](repeating: 0, count: count)
    var n = Int32(count)
    vvexpf(&output, logits, &n)
    var sum: Float = 0
    vDSP_sve(output, 1, &sum, vDSP_Length(count))
    if sum > 0 {
      vDSP_vsdiv(output, 1, &sum, &output, 1, vDSP_Length(count))
    }

    let sorted = output.enumerated().sorted { $0.element > $1.element }
    let top = sorted.prefix(5).filter { $0.offset < labels.count }
    return Probs(
      top1: top.first.map { labels[$0.offset] } ?? "",
      top5: top.map { labels[$0.offset] },
      top1Conf: top.first?.element ?? 0,
      top5Confs: top.map { $0.element }
    )
  }
}
