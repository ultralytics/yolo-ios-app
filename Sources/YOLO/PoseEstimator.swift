// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, implementing human pose estimation functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The PoseEstimator class extends the BasePredictor to provide human pose and keypoint detection.
//  It processes model outputs to identify human subjects and their body keypoints (joints such as
//  eyes, shoulders, elbows, wrists, hips, knees, ankles, etc.). The class converts the model's raw
//  output into structured data representing each detected person's bounding box and associated
//  keypoints with their confidence scores. This implementation supports both real-time processing
//  for camera feeds and single image analysis, producing visualizable results that can be overlaid
//  on the source image to show the detected pose skeleton.

import Accelerate
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO pose estimation models that identify human body keypoints.
class PoseEstimator: BasePredictor, @unchecked Sendable {
  var colorsForMask: [(red: UInt8, green: UInt8, blue: UInt8)] = []

  override func processObservations(for request: VNRequest, error: Error?) {
    if let results = request.results as? [VNCoreMLFeatureValueObservation] {

      if let prediction = results.first?.featureValue.multiArrayValue {

        let preds = PostProcessPose(
          prediction: prediction, confidenceThreshold: Float(self.confidenceThreshold),
          iouThreshold: Float(self.iouThreshold))
        var keypointsList = [Keypoints]()
        var boxes = [Box]()

        let limitedPreds = preds.prefix(self.numItemsThreshold)
        for person in limitedPreds {
          boxes.append(person.box)
          keypointsList.append(person.keypoints)
        }
        let result = YOLOResult(
          orig_shape: inputSize, boxes: boxes, masks: nil, probs: nil, keypointsList: keypointsList,
          annotatedImage: nil, speed: 0, fps: 0, originalImage: nil, names: labels)
        self.currentOnResultsListener?.on(result: result)
        self.updateTime()
      }
    }
  }

  private func updateTime() {
    if self.t1 < 10.0 {  // valid dt
      self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
    }
    self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
    self.t3 = CACurrentMediaTime()

    self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)  // t2 seconds to ms

  }

  override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
      return emptyResult
    }

    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    let result = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)

    do {
      try requestHandler.perform([request])

      if let results = request.results as? [VNCoreMLFeatureValueObservation] {

        if let prediction = results.first?.featureValue.multiArrayValue {

          let preds = PostProcessPose(
            prediction: prediction, confidenceThreshold: Float(self.confidenceThreshold),
            iouThreshold: Float(self.iouThreshold))
          var keypointsList = [Keypoints]()
          var boxes = [Box]()
          var keypointsForImage = [[(x: Float, y: Float)]]()
          var confsList: [[Float]] = []

          let limitedPreds = preds.prefix(self.numItemsThreshold)
          for person in limitedPreds {
            boxes.append(person.box)
            keypointsList.append(person.keypoints)
            keypointsForImage.append(person.keypoints.xyn)
            confsList.append(person.keypoints.conf)
          }

          // 新しい統合描画関数を使用
          let annotatedImage = drawYOLOPoseWithBoxes(
            ciImage: image,
            keypointsList: keypointsForImage,
            confsList: confsList,
            boundingBoxes: boxes,
            originalImageSize: inputSize
          )

          let result = YOLOResult(
            orig_shape: inputSize, boxes: boxes, masks: nil, probs: nil,
            keypointsList: keypointsList, annotatedImage: annotatedImage, speed: self.t2,
            fps: 1 / self.t4, originalImage: nil, names: labels)
          updateTime()
          return result
        }
      }
    } catch {
      print(error)
    }
    return result
  }

  func PostProcessPose(
    prediction: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  )
    -> [(box: Box, keypoints: Keypoints)]
  {
    let numAnchors = prediction.shape[2].intValue
    let featureCount = prediction.shape[1].intValue - 5

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
        let boxX = CGFloat(x - width / 2.0)
        let boxY = CGFloat(y - height / 2.0)
        let boundingBox = CGRect(
          x: boxX, y: boxY,
          width: boxWidth, height: boxHeight)

        var boxFeatures = [Float](repeating: 0, count: featureCount)
        for k in 0..<featureCount {
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

    let filteredBoxes = selectedIndices.map { boxes[$0] }
    let filteredScores = selectedIndices.map { scores[$0] }
    let filteredFeatures = selectedIndices.map { features[$0] }

    let boxScorePairs = zip(filteredBoxes, filteredScores)
    let results: [(Box, Keypoints)] = zip(boxScorePairs, filteredFeatures).map {
      (pair, boxFeatures) in
      let (box, score) = pair
      let Nx = box.origin.x / CGFloat(modelInputSize.width)
      let Ny = box.origin.y / CGFloat(modelInputSize.height)
      let Nw = box.size.width / CGFloat(modelInputSize.width)
      let Nh = box.size.height / CGFloat(modelInputSize.height)
      let ix = Nx * inputSize.width
      let iy = Ny * inputSize.height
      let iw = Nw * inputSize.width
      let ih = Nh * inputSize.height
      let normalizedBox = CGRect(x: Nx, y: Ny, width: Nw, height: Nh)
      let imageSizeBox = CGRect(x: ix, y: iy, width: iw, height: ih)
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

        let x = nX * Float(inputSize.width)
        let y = nY * Float(inputSize.height)
        xyArray.append((x: x, y: y))

        confArray.append(kc)
      }

      let keypoints = Keypoints(xyn: xynArray, xy: xyArray, conf: confArray)
      return (boxResult, keypoints)
    }

    return results
  }
}
