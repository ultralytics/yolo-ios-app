// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import Accelerate
import CoreML
import Foundation
import Vision

/// Specialized predictor for YOLO pose estimation models.
public class PoseEstimator: BasePredictor {

  override func processResults() -> YOLOResult {
    guard let request = visionRequest,
      let results = request.results as? [VNCoreMLFeatureValueObservation],
      let prediction = results.first?.featureValue.multiArrayValue
    else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
    }

    let preds = postProcessPose(
      prediction: prediction,
      confidenceThreshold: Float(configuration.confidenceThreshold),
      iouThreshold: Float(configuration.iouThreshold))

    var keypointsList = [Keypoints]()
    var boxes = [Box]()
    let limitedPreds = preds.prefix(configuration.maxDetections)
    for person in limitedPreds {
      boxes.append(person.box)
      keypointsList.append(person.keypoints)
    }

    return YOLOResult(
      orig_shape: inputSize, boxes: boxes, keypointsList: keypointsList,
      speed: t1, fps: t4 > 0 ? 1 / t4 : nil, names: labels)
  }

  private func postProcessPose(
    prediction: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(box: Box, keypoints: Keypoints)] {
    let numAnchors = prediction.shape[2].intValue
    let featureCount = prediction.shape[1].intValue - 5

    final class CollectionsWrapper: @unchecked Sendable {
      private let lock = NSLock()
      private var boxes: [CGRect] = []
      private var scores: [Float] = []
      private var features: [[Float]] = []

      func append(box: CGRect, score: Float, feature: [Float]) {
        lock.lock()
        boxes.append(box)
        scores.append(score)
        features.append(feature)
        lock.unlock()
      }

      func getCollections() -> (boxes: [CGRect], scores: [Float], features: [[Float]]) {
        (boxes, scores, features)
      }
    }

    struct PointerWrapper: @unchecked Sendable {
      let pointer: UnsafeMutablePointer<Float>
    }

    let featurePointer = UnsafeMutablePointer<Float>(OpaquePointer(prediction.dataPointer))
    let pointerWrapper = PointerWrapper(pointer: featurePointer)
    let collectionsWrapper = CollectionsWrapper()

    DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
      let confIndex = 4 * numAnchors + j
      let confidence = pointerWrapper.pointer[confIndex]
      if confidence > confidenceThreshold {
        let x = pointerWrapper.pointer[j]
        let y = pointerWrapper.pointer[numAnchors + j]
        let width = pointerWrapper.pointer[2 * numAnchors + j]
        let height = pointerWrapper.pointer[3 * numAnchors + j]
        let boundingBox = CGRect(
          x: CGFloat(x - width / 2), y: CGFloat(y - height / 2),
          width: CGFloat(width), height: CGFloat(height))

        var boxFeatures = [Float](repeating: 0, count: featureCount)
        for k in 0..<featureCount {
          boxFeatures[k] = pointerWrapper.pointer[(5 + k) * numAnchors + j]
        }
        collectionsWrapper.append(box: boundingBox, score: confidence, feature: boxFeatures)
      }
    }

    let collections = collectionsWrapper.getCollections()
    let selectedIndices = nonMaxSuppression(
      boxes: collections.boxes, scores: collections.scores, threshold: iouThreshold)

    let filteredBoxes = selectedIndices.map { collections.boxes[$0] }
    let filteredScores = selectedIndices.map { collections.scores[$0] }
    let filteredFeatures = selectedIndices.map { collections.features[$0] }

    return zip(zip(filteredBoxes, filteredScores), filteredFeatures).map { pair, boxFeatures in
      let (box, score) = pair
      let Nx = box.origin.x / CGFloat(modelInputSize.width)
      let Ny = box.origin.y / CGFloat(modelInputSize.height)
      let Nw = box.size.width / CGFloat(modelInputSize.width)
      let Nh = box.size.height / CGFloat(modelInputSize.height)
      let normalizedBox = CGRect(x: Nx, y: Ny, width: Nw, height: Nh)
      let imageSizeBox = CGRect(
        x: Nx * inputSize.width, y: Ny * inputSize.height,
        width: Nw * inputSize.width, height: Nh * inputSize.height)
      let boxResult = Box(
        index: 0, cls: "person", conf: score, xywh: imageSizeBox, xywhn: normalizedBox)

      let numKeypoints = boxFeatures.count / 3
      var xynArray = [(x: Float, y: Float)]()
      var xyArray = [(x: Float, y: Float)]()
      var confArray = [Float]()

      for i in 0..<numKeypoints {
        let kx = boxFeatures[3 * i]
        let ky = boxFeatures[3 * i + 1]
        let kc = boxFeatures[3 * i + 2]
        let nX = kx / Float(modelInputSize.width)
        let nY = ky / Float(modelInputSize.height)
        xynArray.append((x: nX, y: nY))
        xyArray.append((x: nX * Float(inputSize.width), y: nY * Float(inputSize.height)))
        confArray.append(kc)
      }

      return (boxResult, Keypoints(xyn: xynArray, xy: xyArray, conf: confArray))
    }
  }
}
