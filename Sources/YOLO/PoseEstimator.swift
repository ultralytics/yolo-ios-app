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
public class PoseEstimator: BasePredictor, @unchecked Sendable {
  var colorsForMask: [(red: UInt8, green: UInt8, blue: UInt8)] = []

  /// Checks if the current model is a YOLO26 model
  private var isYOLO26Model: Bool { isYOLO26Model(from: modelURL) }

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

  public override func predictOnImage(image: CIImage) -> YOLOResult {
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
    let shape = prediction.shape.map { $0.intValue }

    // YOLO26 pose post-NMS: [batch, num_detections, features] or [batch, features, num_detections]
    // features = 6 (box+conf+class) + num_keypoints*3; e.g. 57 for 17 keypoints. Anchor format: [1, 56, 8400].
    // Only use YOLO26 path when shape looks like post-NMS: small dim = 6+3*k, large dim < 5000.
    if isYOLO26Model && shape.count == 3 {
      let s = min(shape[1], shape[2])
      let l = max(shape[1], shape[2])
      if s >= 9 && (s - 6) % 3 == 0 && l < 5000 {
        let detectionFirst = shape[1] > shape[2]
        let numDetections = detectionFirst ? shape[1] : shape[2]
        let numFeatures = detectionFirst ? shape[2] : shape[1]
        return postProcessYOLO26PoseFormat(
          feature: prediction,
          numDetections: numDetections,
          numFeatures: numFeatures,
          detectionFirst: detectionFirst,
          confidenceThreshold: confidenceThreshold,
          iouThreshold: iouThreshold,
          modelInputSize: self.modelInputSize,
          inputSize: self.inputSize
        )
      }
    }

    // YOLO11 anchor-based format: [batch, features, anchors]
    let numAnchors = shape.count >= 3 ? shape[2] : shape[1]
    let featureCount = shape.count >= 3 ? (shape[1] - 5) : (shape[0] - 5)

    var boxes = [CGRect]()
    var scores = [Float]()
    var features = [[Float]]()

    // Wrapper for thread-safe collections
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
        return (boxes, scores, features)
      }
    }

    // Wrapper to make pointer Sendable
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
          boxFeatures[k] = pointerWrapper.pointer[key]
        }

        collectionsWrapper.append(box: boundingBox, score: confidence, feature: boxFeatures)
      }
    }

    // Get collections from wrapper
    let collections = collectionsWrapper.getCollections()
    boxes = collections.boxes
    scores = collections.scores
    features = collections.features

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

  /// Post-processes YOLO26 pose model output in post-NMS format
  /// Layout: detectionFirst ? [batch, num_detections, features] : [batch, features, num_detections]
  /// features = [x1, y1, x2, y2, confidence, class_idx, kx0, ky0, kc0, ...]
  nonisolated func postProcessYOLO26PoseFormat(
    feature: MLMultiArray,
    numDetections: Int,
    numFeatures: Int,
    detectionFirst: Bool,
    confidenceThreshold: Float,
    iouThreshold: Float,
    modelInputSize: (width: Int, height: Int),
    inputSize: CGSize
  ) -> [(box: Box, keypoints: Keypoints)] {
    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)

    func value(detection: Int, featureIndex: Int) -> Float {
      if detectionFirst {
        return featurePointer[detection * numFeatures + featureIndex]
      } else {
        return featurePointer[featureIndex * numDetections + detection]
      }
    }

    let numKeypoints = (numFeatures - 6) / 3
    var detections: [(CGRect, Float, [Float])] = []
    detections.reserveCapacity(min(numDetections, 100))

    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)

    for i in 0..<numDetections {
      let x1 = CGFloat(value(detection: i, featureIndex: 0))
      let y1 = CGFloat(value(detection: i, featureIndex: 1))
      let x2 = CGFloat(value(detection: i, featureIndex: 2))
      let y2 = CGFloat(value(detection: i, featureIndex: 3))
      var confidence = value(detection: i, featureIndex: 4)
      _ = value(detection: i, featureIndex: 5)

      if confidence > 1.0 && confidence <= 100.0 {
        confidence = confidence / 100.0
      } else if confidence > 100.0 {
        confidence = 1.0 / (1.0 + exp(-confidence))
      }

      guard confidence > confidenceThreshold else {
        continue
      }

      let boxX = x1 / modelWidth
      let boxY = y1 / modelHeight
      let boxW = (x2 - x1) / modelWidth
      let boxH = (y2 - y1) / modelHeight

      let clampedX = max(0.0, min(1.0, boxX))
      let clampedY = max(0.0, min(1.0, boxY))
      let clampedW = max(0.0, min(1.0 - clampedX, boxW))
      let clampedH = max(0.0, min(1.0 - clampedY, boxH))

      guard clampedW > 0.01 && clampedH > 0.01 else {
        continue
      }

      let boundingBox = CGRect(x: clampedX, y: clampedY, width: clampedW, height: clampedH)

      var keypointFeatures: [Float] = []
      keypointFeatures.reserveCapacity(numKeypoints * 3)
      for k in 0..<numKeypoints {
        keypointFeatures.append(value(detection: i, featureIndex: 6 + k * 3))
        keypointFeatures.append(value(detection: i, featureIndex: 6 + k * 3 + 1))
        keypointFeatures.append(value(detection: i, featureIndex: 6 + k * 3 + 2))
      }

      detections.append((boundingBox, confidence, keypointFeatures))
    }

    // Apply NMS (group by class first, then apply NMS per class)
    var classBuckets: [Int: [(CGRect, Float, [Float])]] = [:]
    for detection in detections {
      // For pose models, all detections are typically class 0 (person)
      let classIndex = 0
      if classBuckets[classIndex] == nil {
        classBuckets[classIndex] = []
      }
      classBuckets[classIndex]?.append(detection)
    }

    var selectedDetections: [(CGRect, Float, [Float])] = []
    for (_, classDetections) in classBuckets {
      let boxesOnly = classDetections.map { $0.0 }
      let scoresOnly = classDetections.map { $0.1 }
      let selectedIndices = nonMaxSuppression(
        boxes: boxesOnly,
        scores: scoresOnly,
        threshold: iouThreshold
      )
      for idx in selectedIndices {
        selectedDetections.append(classDetections[idx])
      }
    }

    // Convert to result format
    let results: [(Box, Keypoints)] = selectedDetections.map { (box, score, keypointFeatures) in
      let Nx = box.origin.x
      let Ny = box.origin.y
      let Nw = box.size.width
      let Nh = box.size.height
      let ix = Nx * inputSize.width
      let iy = Ny * inputSize.height
      let iw = Nw * inputSize.width
      let ih = Nh * inputSize.height
      let normalizedBox = CGRect(x: Nx, y: Ny, width: Nw, height: Nh)
      let imageSizeBox = CGRect(x: ix, y: iy, width: iw, height: ih)
      let boxResult = Box(
        index: 0, cls: "person", conf: score, xywh: imageSizeBox, xywhn: normalizedBox)

      var xynArray = [(x: Float, y: Float)]()
      var xyArray = [(x: Float, y: Float)]()
      var confArray = [Float]()

      for i in 0..<numKeypoints {
        let kx = keypointFeatures[i * 3]
        let ky = keypointFeatures[i * 3 + 1]
        let kc = keypointFeatures[i * 3 + 2]

        // Normalize keypoint coordinates from pixel space to 0-1
        let nX = kx / Float(modelWidth)
        let nY = ky / Float(modelHeight)
        xynArray.append((x: nX, y: nY))

        // Convert to image space
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
