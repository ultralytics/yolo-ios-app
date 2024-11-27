//
//  Segment.swift
//  YOLO
//

import Accelerate
import Foundation
import UIKit
import Vision

extension ViewController {
  func setupSegmentPoseOverlay() {
    let width = videoPreview.bounds.width
    let height = videoPreview.bounds.height

    var ratio: CGFloat = 1.0
    if videoCapture.captureSession.sessionPreset == .photo {
      ratio = (4.0 / 3.0)
    } else {
      ratio = (16.0 / 9.0)
    }
    var offSet = CGFloat.zero
    var margin = CGFloat.zero
    if view.bounds.width < view.bounds.height {
      offSet = height / ratio
      margin = (offSet - self.videoPreview.bounds.width) / 2
      self.segmentPoseOverlay.frame = CGRect(
        x: -margin, y: 0, width: offSet, height: self.videoPreview.bounds.height)
    } else {
      offSet = width / ratio
      margin = (offSet - self.videoPreview.bounds.height) / 2
      self.segmentPoseOverlay.frame = CGRect(
        x: 0, y: -margin, width: self.videoPreview.bounds.width, height: offSet)

    }
    var count = 0
    for _ in colors {
      let color = ultralyticsColorsolors[count]
      count += 1
      if count > 19 {
        count = 0
      }
      guard let colorForMask = color.toRGBComponents() else { fatalError() }
      colorsForMasks.append(colorForMask)
    }
  }

  func postProcessSegment(request: VNRequest) {
    if let results = request.results as? [VNCoreMLFeatureValueObservation] {
      DispatchQueue.main.async { [self] in
        guard results.count == 2 else { return }
        let masks = results[0].featureValue.multiArrayValue
        let pred = results[1].featureValue.multiArrayValue
        let a = Date()

        let processed = getBoundingBoxesAndMasks(
          feature: pred!, confidenceThreshold: 0.25, iouThreshold: 0.4)
        var predictions = [DetectionResult]()
        for object in processed {
          let box = object.0
          let rect = CGRect(
            x: box.minX / 640, y: box.minY / 640, width: box.width / 640, height: box.height / 640)
          let bestClass = classes[object.1]
          let confidence = object.2
          let prediction = DetectionResult(rect: rect, label: bestClass, confidence: confidence)
          predictions.append(prediction)
        }
        self.showBoundingBoxes(predictions: predictions)
        self.updateMaskAndBoxes(detectedObjects: processed, maskArray: masks!)
      }
    }
  }

  func getBoundingBoxesAndMasks(
    feature: MLMultiArray, confidenceThreshold: Float, iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let numAnchors = feature.shape[2].intValue
    let numFeatures = feature.shape[1].intValue
    let boxFeatureLength = 4
    let maskConfidenceLength = 32
    let numClasses = numFeatures - boxFeatureLength - maskConfidenceLength

    var results = [(CGRect, Int, Float, MLMultiArray)]()
    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)

    let resultsQueue = DispatchQueue(label: "resultsQueue", attributes: .concurrent)

    DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
      let baseOffset = j
      let x = featurePointer[baseOffset]
      let y = featurePointer[numAnchors + baseOffset]
      let width = featurePointer[2 * numAnchors + baseOffset]
      let height = featurePointer[3 * numAnchors + baseOffset]

      let boxWidth = CGFloat(width)
      let boxHeight = CGFloat(height)
      let boxX = CGFloat(x - width / 2)
      let boxY = CGFloat(y - height / 2)

      let boundingBox = CGRect(x: boxX, y: boxY, width: boxWidth, height: boxHeight)

      var classProbs = [Float](repeating: 0, count: numClasses)
      classProbs.withUnsafeMutableBufferPointer { classProbsPointer in
        vDSP_mtrans(
          featurePointer + 4 * numAnchors + baseOffset, numAnchors, classProbsPointer.baseAddress!,
          1, 1, vDSP_Length(numClasses))
      }
      var maxClassValue: Float = 0
      var maxClassIndex: vDSP_Length = 0
      vDSP_maxvi(classProbs, 1, &maxClassValue, &maxClassIndex, vDSP_Length(numClasses))

      if maxClassValue > confidenceThreshold {
        let maskProbsPointer = featurePointer + (4 + numClasses) * numAnchors + baseOffset
        let maskProbs = try! MLMultiArray(
          shape: [NSNumber(value: maskConfidenceLength)], dataType: .float32)
        for i in 0..<maskConfidenceLength {
          maskProbs[i] = NSNumber(value: maskProbsPointer[i * numAnchors])
        }

        let result = (boundingBox, Int(maxClassIndex), maxClassValue, maskProbs)

        // Using resultsQueue to synchronize access to results
        resultsQueue.async(flags: .barrier) {
          results.append(result)
        }
      }
    }

    // Ensuring all updates to results are complete
    resultsQueue.sync(flags: .barrier) {}

    var selectedBoxesAndFeatures = [(CGRect, Int, Float, MLMultiArray)]()

    // Perform NMS class by class
    for classIndex in 0..<numClasses {
      let classResults = results.filter { $0.1 == classIndex }
      if !classResults.isEmpty {
        let boxesOnly = classResults.map { $0.0 }
        let scoresOnly = classResults.map { $0.2 }
        let selectedIndices = nonMaxSuppression(
          boxes: boxesOnly, scores: scoresOnly, threshold: iouThreshold)
        for idx in selectedIndices {
          selectedBoxesAndFeatures.append(
            (classResults[idx].0, classResults[idx].1, classResults[idx].2, classResults[idx].3))
        }
      }
    }

    return selectedBoxesAndFeatures
  }

  private var isUpdating: Bool {
    get {
      return objc_getAssociatedObject(self, &AssociatedKeys.isUpdating) as? Bool ?? false
    }
    set {
      objc_setAssociatedObject(
        self, &AssociatedKeys.isUpdating, newValue, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
  }

  struct AssociatedKeys {
    static var isUpdating = "isUpdating"
  }

  func updateMaskAndBoxes(
    detectedObjects: [(CGRect, Int, Float, MLMultiArray)], maskArray: MLMultiArray
  ) {
    // 実行中ならスキップ
    guard !isUpdating else {
      print("Skipping updateMaskAndBoxes because it is already running")
      return
    }

    // 実行中フラグをセット
    isUpdating = true

    if detectedObjects.isEmpty {
      DispatchQueue.main.async {
        self.removeAllMaskSubLayers()
        self.isUpdating = false  // フラグを解除
      }
      return
    }

    let startTime = Date()
    let group = DispatchGroup()

    let sortedObjects = detectedObjects.sorted {
      $0.0.size.width * $0.0.size.height > $1.0.size.width * $1.0.size.height
    }

    var newLayers: [CALayer] = []

    for (box, classIndex, _, masksIn) in sortedObjects {
      group.enter()
      DispatchQueue.global(qos: .userInitiated).async {
        if let maskImage = self.generateColoredMaskImage(
          from: masksIn, protos: maskArray, in: self.segmentPoseOverlay.bounds.size,
          colorIndex: classIndex,
          boundingBox: box)
        {
          let adjustedBox = self.adjustBox(box, toFitIn: self.segmentPoseOverlay.bounds.size)

          let maskImageLayer = CALayer()
          maskImageLayer.frame = adjustedBox
          maskImageLayer.contents = maskImage
          maskImageLayer.opacity = 0.5
          DispatchQueue.main.async {
            newLayers.append(maskImageLayer)
          }
        }
        group.leave()
      }
    }

    // 全タスクの終了を待つ
    group.notify(queue: .main) { [weak self] in
      guard let self = self else { return }
      self.removeAllMaskSubLayers()
      newLayers.forEach { self.segmentPoseOverlay.addSublayer($0) }
      print("update complete")
      print("Time elapsed: \(Date().timeIntervalSince(startTime))")
      self.isUpdating = false  // フラグを解除
    }
  }
  func generateColoredMaskImage(
    from masksIn: MLMultiArray, protos: MLMultiArray, in size: CGSize, colorIndex: Int,
    boundingBox: CGRect
  ) -> CGImage? {
    let maskWidth = protos.shape[3].intValue
    let maskHeight = protos.shape[2].intValue
    let maskChannels = protos.shape[1].intValue

    guard protos.shape.count == 4, protos.shape[0].intValue == 1, masksIn.shape.count == 1,
      masksIn.shape[0].intValue == maskChannels
    else {
      print("Invalid shapes for protos or masksIn")
      return nil
    }

    let masksPointer = masksIn.dataPointer.assumingMemoryBound(to: Float.self)
    let protosPointer = protos.dataPointer.assumingMemoryBound(to: Float.self)

    let masksPointerOutput = UnsafeMutablePointer<Float>.allocate(capacity: maskHeight * maskWidth)
    vDSP_mmul(
      masksPointer, 1, protosPointer, 1, masksPointerOutput, 1, vDSP_Length(1),
      vDSP_Length(maskHeight * maskWidth), vDSP_Length(maskChannels))

    let threshold: Float = 0.5
    let color = colorsForMask[colorIndex]
    let red = UInt8(color.red)
    let green = UInt8(color.green)
    let blue = UInt8(color.blue)

    var maskPixels = [UInt8](repeating: 0, count: maskHeight * maskWidth * 4)
    for y in 0..<maskHeight {
      for x in 0..<maskWidth {
        let index = y * maskWidth + x
        let maskValue = masksPointerOutput[index]
        if maskValue > threshold {
          let pixelIndex = index * 4
          maskPixels[pixelIndex] = red
          maskPixels[pixelIndex + 1] = green
          maskPixels[pixelIndex + 2] = blue
          maskPixels[pixelIndex + 3] = 255
        }
      }
    }

    let maskDataPointer = UnsafeMutablePointer<UInt8>.allocate(capacity: maskPixels.count)
    maskDataPointer.initialize(from: maskPixels, count: maskPixels.count)

    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    let colorSpace = CGColorSpaceCreateDeviceRGB()

    let maskDataProvider = CGDataProvider(
      dataInfo: nil, data: maskDataPointer, size: maskPixels.count
    ) { _, data, _ in
      data.deallocate()
    }

    guard
      let maskCGImage = CGImage(
        width: maskWidth, height: maskHeight, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: maskWidth * 4, space: colorSpace, bitmapInfo: bitmapInfo,
        provider: maskDataProvider!, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    else {
      masksPointerOutput.deallocate()
      return nil
    }

    let maskCIImage = CIImage(cgImage: maskCGImage)
    let scaledCIImage = maskCIImage.transformed(
      by: CGAffineTransform(
        scaleX: size.width / CGFloat(maskWidth), y: size.height / CGFloat(maskHeight)))
    let invertedY = size.height - (boundingBox.origin.y + boundingBox.height) * size.height / 640.0
    let cropRect = CGRect(
      x: boundingBox.origin.x * size.width / 640.0, y: invertedY,
      width: boundingBox.width * size.width / 640.0,
      height: boundingBox.height * size.height / 640.0)

    let croppedCIImage = scaledCIImage.cropped(to: cropRect)

    let ciContext = CIContext()
    guard let cgImage = ciContext.createCGImage(croppedCIImage, from: cropRect) else {
      masksPointerOutput.deallocate()
      return nil
    }

    masksPointerOutput.deallocate()

    return cgImage
  }

  func removeAllMaskSubLayers() {
    self.segmentPoseOverlay.sublayers?.forEach { layer in
      layer.removeFromSuperlayer()
    }
    self.segmentPoseOverlay.sublayers = nil
  }

  func adjustBox(_ box: CGRect, toFitIn containerSize: CGSize) -> CGRect {
    let xScale = containerSize.width / 640.0
    let yScale = containerSize.height / 640.0
    return CGRect(
      x: box.origin.x * xScale, y: box.origin.y * yScale, width: box.size.width * xScale,
      height: box.size.height * yScale)
  }
}

extension UIColor {
  func toRGBComponents() -> (red: UInt8, green: UInt8, blue: UInt8)? {
    var red: CGFloat = 0
    var green: CGFloat = 0
    var blue: CGFloat = 0
    var alpha: CGFloat = 0

    let success = self.getRed(&red, green: &green, blue: &blue, alpha: &alpha)

    if success {
      let redUInt8 = UInt8(red * 255.0)
      let greenUInt8 = UInt8(green * 255.0)
      let blueUInt8 = UInt8(blue * 255.0)
      return (red: redUInt8, green: greenUInt8, blue: blueUInt8)
    } else {
      return nil
    }
  }
}

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

extension CGRect {
  var area: CGFloat {
    return width * height
  }
}
