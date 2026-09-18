// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO SDK, implementing image classification.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  Classifier identifies the primary subject of an image rather than locating objects within it. It accepts both
//  raw probability tensors (Core AI, and Core ML models without a classifier config) and
//  `VNClassificationObservation` results, and reports the top-1 plus top-5 predictions with confidence scores.

import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO classification models that identify the subject of an image.
public final class Classifier: BasePredictor, @unchecked Sendable {

  override var imageCropAndScaleOption: VNImageCropAndScaleOption { .centerCrop }

  override func processObservations(for request: VNRequest, _ error: Error?) {
    markInferenceEnd()
    let probs = extractProbs(from: request)
    self.updateTime()
    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: self.t2, fps: 1 / self.t4,
      names: labels)
    applyTimingBreakdown(&result, smoothed: true)
    result.originalImage = currentOriginalImage
    self.currentOnResultsListener?.on(result: result)
  }

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    var probs = Probs(top1: "", top5: [], top1Conf: 0, top5Confs: [])
    let requestHandler = makeRequestHandler(for: image)
    if perform(request, with: requestHandler, errorMessage: "Classifier inference failed") {
      markInferenceEnd()
      probs = extractProbs(from: request)
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: 0, names: labels)
    result.speed = finishTiming(notify: false)  // before drawing: annotation is excluded from timings
    applyTimingBreakdown(&result)
    result.annotatedImage = drawYOLOClassifications(on: image, result: result)
    if capturesOriginalImage {
      result.originalImage = UIImage(ciImage: image)
    }
    return result
  }

  /// Extracts top-1 and top-5 probabilities from a completed request, handling both a raw probability tensor and
  /// `VNClassificationObservation` results. Ultralytics classify exports apply softmax inside the model.
  private func extractProbs(from request: VNRequest) -> Probs {
    if let multiArray = featureArrays(request).first {
      return topProbs(from: multiArray)
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

  /// Returns the top-1/top-5 classes of a probability tensor.
  func topProbs(from multiArray: MLMultiArray) -> Probs {
    let count = multiArray.count
    var output = [Float](repeating: 0, count: count)
    if multiArray.dataType == .float32, multiArray.strides.last?.intValue == 1 {
      let src = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
      output.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: src, count: count) }
    } else {
      for i in 0..<count { output[i] = multiArray[i].floatValue }
    }

    // Select the top-5 with a single linear pass and a tiny sorted insertion buffer instead of sorting the whole
    // vector. For a 1000-class head this avoids an O(n log n) sort and the enumerated() tuple-array allocation
    // every frame. Equal scores resolve to the lower class index (deterministic; exact ties don't occur for real
    // model outputs).
    let k = min(5, count)
    var topIdx = [Int](repeating: -1, count: k)
    var topVal = [Float](repeating: -.greatestFiniteMagnitude, count: k)
    for i in 0..<count {
      let v = output[i]
      if v <= topVal[k - 1] { continue }
      var p = k - 1
      while p > 0 && v > topVal[p - 1] {
        topVal[p] = topVal[p - 1]
        topIdx[p] = topIdx[p - 1]
        p -= 1
      }
      topVal[p] = v
      topIdx[p] = i
    }
    var topLabels = [String]()
    var topConfs = [Float]()
    for j in 0..<k where topIdx[j] >= 0 {
      topLabels.append(labelName(for: topIdx[j]))
      topConfs.append(topVal[j])
    }
    return Probs(
      top1: topLabels.first ?? "",
      top5: topLabels,
      top1Conf: topConfs.first ?? 0,
      top5Confs: topConfs
    )
  }
}
