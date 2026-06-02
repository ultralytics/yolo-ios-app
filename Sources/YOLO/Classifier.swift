// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO SDK, implementing image classification.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  Classifier identifies the primary subject of an image rather than locating objects within it. It accepts both
//  `VNCoreMLFeatureValueObservation` (raw logits requiring softmax) and `VNClassificationObservation` (already
//  normalized) result types, and reports the top-1 plus top-5 predictions with confidence scores.

import Accelerate
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO classification models that identify the subject of an image.
public final class Classifier: BasePredictor, @unchecked Sendable {

  override var imageCropAndScaleOption: VNImageCropAndScaleOption { .centerCrop }

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
    self.t0 = CACurrentMediaTime()
    do {
      try requestHandler.perform([request])
      probs = extractProbs(from: request)
    } catch {
      YOLOLog.error("Classifier inference failed: \(error)")
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: self.t1, names: labels)
    result.annotatedImage = drawYOLOClassifications(on: image, result: result)
    updateTime(notify: false)
    result.speed = self.t1
    return result
  }

  /// Extracts top-1 and top-5 probabilities from a Vision request result, handling both
  /// `VNCoreMLFeatureValueObservation` (raw logits requiring softmax) and `VNClassificationObservation` (already
  /// normalized scores).
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
  func softmaxProbs(from multiArray: MLMultiArray) -> Probs {
    let count = multiArray.count
    var logits = [Float](repeating: 0, count: count)
    if multiArray.dataType == .float32, multiArray.strides.last?.intValue == 1 {
      let src = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
      logits.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: src, count: count) }
    } else {
      for i in 0..<count { logits[i] = multiArray[i].floatValue }
    }

    var output = [Float](repeating: 0, count: count)
    var maxLogit: Float = 0
    vDSP_maxv(logits, 1, &maxLogit, vDSP_Length(count))
    var negMax = -maxLogit
    vDSP_vsadd(logits, 1, &negMax, &output, 1, vDSP_Length(count))
    var n = Int32(count)
    vvexpf(&output, output, &n)
    var sum: Float = 0
    vDSP_sve(output, 1, &sum, vDSP_Length(count))
    if sum > 0 {
      vDSP_vsdiv(output, 1, &sum, &output, 1, vDSP_Length(count))
    }

    // Select the top-5 with a single linear pass and a tiny sorted insertion buffer instead of sorting the whole
    // vector. For a 1000-class head this avoids an O(n log n) sort and the enumerated() tuple-array allocation
    // every frame. Equal scores resolve to the lower class index (deterministic; exact ties don't occur for real
    // softmax outputs).
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
    for j in 0..<k where topIdx[j] >= 0 && topIdx[j] < labels.count {
      topLabels.append(labels[topIdx[j]])
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
