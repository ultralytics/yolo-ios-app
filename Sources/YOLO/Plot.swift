// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing visualization utilities.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Plot module provides visualization utilities for rendering YOLO model results.
//  It includes functions for drawing bounding boxes, segmentation masks, pose keypoints,
//  classification results, and oriented bounding boxes on images. The module implements
//  specialized rendering algorithms for each type of prediction, handles color management
//  for different classes, and supports both static image and real-time visualization scenarios.
//  Each visualization function is optimized for the specific task to provide clear and
//  informative visual feedback to users with minimal performance impact.

import Accelerate
import CoreImage
import CoreML
import Foundation
import QuartzCore
import UIKit

let ultralyticsColors: [UIColor] = [
  UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 0.6),
  UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 0.6),
  UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 0.6),
  UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 0.6),
  UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 0.6),
  UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 0.6),
  UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 0.6),
  UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 0.6),
  UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 0.6),
  UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 0.6),
  UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 0.6),
  UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 0.6),
  UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 0.6),
  UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 0.6),
  UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 0.6),
]

let posePalette: [[CGFloat]] = [
  [255, 128, 0],
  [255, 153, 51],
  [255, 178, 102],
  [230, 230, 0],
  [255, 153, 255],
  [153, 204, 255],
  [255, 102, 255],
  [255, 51, 255],
  [102, 178, 255],
  [51, 153, 255],
  [255, 153, 153],
  [255, 102, 102],
  [255, 51, 51],
  [153, 255, 153],
  [102, 255, 102],
  [51, 255, 51],
  [0, 255, 0],
  [0, 0, 255],
  [255, 0, 0],
  [255, 255, 255],
]

let limbColorIndices = [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
let kptColorIndices = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

let skeleton = [
  [16, 14],
  [14, 12],
  [17, 15],
  [15, 13],
  [12, 13],
  [6, 12],
  [7, 13],
  [6, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [9, 11],
  [2, 3],
  [1, 2],
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 6],
  [5, 7],
]

public func drawYOLODetections(on ciImage: CIImage, result: YOLOResult) -> UIImage {
  let context = CIContext(options: nil)
  guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
    return UIImage()
  }
  let width = cgImage.width
  let height = cgImage.height
  let imageSize = CGSize(width: width, height: height)
  UIGraphicsBeginImageContextWithOptions(imageSize, false, 1.0)
  guard let drawContext = UIGraphicsGetCurrentContext() else {
    UIGraphicsEndImageContext()
    return UIImage()
  }
  drawContext.saveGState()
  drawContext.translateBy(x: 0, y: CGFloat(height))
  drawContext.scaleBy(x: 1, y: -1)
  drawContext.draw(cgImage, in: CGRect(origin: .zero, size: imageSize))
  drawContext.restoreGState()

  // Calculate line width and font size proportionally to image dimensions
  let lineWidth = max(width, height) / 200
  let fontSize = max(width, height) / 50

  for box in result.boxes {
    let colorIndex = box.index % ultralyticsColors.count
    let color = ultralyticsColors[colorIndex]
    drawContext.setStrokeColor(color.cgColor)
    drawContext.setLineWidth(CGFloat(lineWidth))
    let rect = box.xywh
    drawContext.stroke(rect)
    let confidencePercent = Int(box.conf * 100)
    let labelText = "\(box.cls) \(confidencePercent)%"
    let font = UIFont.systemFont(ofSize: CGFloat(fontSize), weight: .semibold)
    let attrs: [NSAttributedString.Key: Any] = [
      .font: font,
      .foregroundColor: UIColor.white,
    ]
    let textSize = labelText.size(withAttributes: attrs)
    let labelWidth = textSize.width + 10
    let labelHeight = textSize.height + 4
    var labelRect = CGRect(
      x: rect.minX,
      y: rect.minY - labelHeight,
      width: labelWidth,
      height: labelHeight
    )
    if labelRect.minY < 0 {
      labelRect.origin.y = rect.minY
    }
    drawContext.setFillColor(color.cgColor)
    drawContext.fill(labelRect)
    let textPoint = CGPoint(
      x: labelRect.origin.x + 5,
      y: labelRect.origin.y + (labelHeight - textSize.height) / 2
    )
    labelText.draw(at: textPoint, withAttributes: attrs)
  }
  let drawnImage = UIGraphicsGetImageFromCurrentImageContext() ?? UIImage()
  UIGraphicsEndImageContext()
  return drawnImage
}

func generateCombinedMaskImage(
  detectedObjects: [(CGRect, Int, Float, MLMultiArray)],
  protos: MLMultiArray,  // shape: [1, C, H, W]
  inputWidth: Int,
  inputHeight: Int,
  threshold: Float = 0.5,
  returnIndividualMasks: Bool = true
) -> (CGImage?, [[[Float]]]?)? {
  // 1) Check protos shape
  let maskHeight = protos.shape[2].intValue  // 例: 160
  let maskWidth = protos.shape[3].intValue  // 例: 160
  let maskChannels = protos.shape[1].intValue  // 例: 32
  guard
    protos.shape.count == 4,
    protos.shape[0].intValue == 1,
    maskHeight > 0,
    maskWidth > 0,
    maskChannels > 0
  else {
    print("Invalid protos shape!")
    return nil
  }

  let protosPointer = protos.dataPointer.assumingMemoryBound(to: Float.self)
  let HW = maskHeight * maskWidth
  let N = detectedObjects.count

  // 2) Prepare matrix A: (N, C) at once (number of objects x mask channels)
  var coeffsArray = [Float](repeating: 0, count: N * maskChannels)
  for i in 0..<N {
    let (_, _, _, coeffsMLArray) = detectedObjects[i]
    let coeffsPtr = coeffsMLArray.dataPointer.assumingMemoryBound(to: Float.self)
    // Row i of matrix A: write to coeffsArray[i*C .. i*C + C-1]
    for c in 0..<maskChannels {
      coeffsArray[i * maskChannels + c] = coeffsPtr[c]
    }
  }

  // 3) Matrix B: (C, HW) uses protosPointer directly
  //    Memory layout is [1, C, H, W] => (C, H, W) => (C, HW). Rows: C, Columns: HW
  //    vDSP_mmul simply treats contiguous memory as 2D, so this is OK.

  // 4) Matrix C (output): (N, HW) allocate => combinedMask
  //    A flat 1D array with N*HW elements
  var combinedMask = [Float](repeating: 0, count: N * HW)

  // 5) Batch computation with vDSP_mmul: (N x C) * (C x HW) => (N x HW)
  coeffsArray.withUnsafeBufferPointer { Abuf in
    combinedMask.withUnsafeMutableBufferPointer { Cbuf in
      vDSP_mmul(
        Abuf.baseAddress!, 1,  // A
        protosPointer, 1,  // B
        Cbuf.baseAddress!, 1,  // C
        vDSP_Length(N),
        vDSP_Length(HW),
        vDSP_Length(maskChannels)
      )
    }
  }

  // 6) Sort by score (to control drawing order during composition)
  //    => (originalIndex, box, classID, score)
  let indexedObjects: [(Int, CGRect, Int, Float)] =
    detectedObjects.enumerated().map { (i, obj) in (i, obj.0, obj.1, obj.2) }
  let sortedObjects = indexedObjects.sorted { $0.3 < $1.3 }  // ascending by score

  // 7) RGBA buffer (160x160)
  var mergedPixels = [UInt8](repeating: 0, count: HW * 4)
  let scaleX = Float(maskWidth) / Float(inputWidth)
  let scaleY = Float(maskHeight) / Float(inputHeight)

  // 8) Whether to keep individual probability maps
  var probabilityMasks: [[[Float]]]? = nil
  if returnIndividualMasks {
    probabilityMasks = Array(
      repeating: Array(
        repeating: Array(repeating: Float(0.0), count: maskWidth),
        count: maskHeight
      ),
      count: N
    )
  }

  // 9) Compose according to sort order
  for (originalIndex, box, classID, _) in sortedObjects {
    // Convert boundingBox to mask coordinate system
    let minX = Int(Float(box.minX) * scaleX)
    let minY = Int(Float(box.minY) * scaleY)
    let maxX = Int(Float(box.maxX) * scaleX)
    let maxY = Int(Float(box.maxY) * scaleY)

    let boxX1 = max(0, min(minX, maskWidth - 1))
    let boxX2 = max(0, min(maxX, maskWidth - 1))
    let boxY1 = max(0, min(minY, maskHeight - 1))
    let boxY2 = max(0, min(maxY, maskHeight - 1))

    let startIdx = originalIndex * HW

    // Get class color
    let _colorIndex = classID % ultralyticsColors.count
    guard let color = ultralyticsColors[_colorIndex].toRGBComponents() else {
      continue
    }
    let r = UInt8(color.red)
    let g = UInt8(color.green)
    let b = UInt8(color.blue)

    // Pixel loop: box range only
    for y in boxY1...boxY2 {
      for x in boxX1...boxX2 {
        let px = y * maskWidth + x
        let maskVal = combinedMask[startIdx + px]
        if maskVal > threshold {
          let pixIndex = px * 4
          mergedPixels[pixIndex + 0] = r
          mergedPixels[pixIndex + 1] = g
          mergedPixels[pixIndex + 2] = b
          mergedPixels[pixIndex + 3] = 255
        }
      }
    }
  }

  if returnIndividualMasks, var masksArray = probabilityMasks {
    for i in 0..<N {
      let startIdx = i * HW
      for k in 0..<HW {
        let row = k / maskWidth
        let col = k % maskWidth
        masksArray[i][row][col] = combinedMask[startIdx + k]
      }
    }
    probabilityMasks = masksArray
  }

  // 11) RGBA buffer -> CGImage
  let colorSpace = CGColorSpaceCreateDeviceRGB()
  let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
  let totalBytes = mergedPixels.count

  guard let providerRef = CGDataProvider(data: NSData(bytes: &mergedPixels, length: totalBytes))
  else {
    return nil
  }
  guard
    let mergedCGImage = CGImage(
      width: maskWidth,
      height: maskHeight,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: maskWidth * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: providerRef,
      decode: nil,
      shouldInterpolate: false,
      intent: .defaultIntent
    )
  else {
    return nil
  }

  return (mergedCGImage, probabilityMasks)
}

func composeImageWithMask(
  baseImage: CGImage,
  maskImage: CGImage
) -> UIImage? {
  let width = baseImage.width
  let height = baseImage.height

  guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
  guard
    let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )
  else {
    return nil
  }

  let baseRect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
  context.draw(baseImage, in: baseRect)

  context.saveGState()
  context.setAlpha(0.5)
  context.draw(maskImage, in: baseRect)
  context.restoreGState()

  guard let composedImage = context.makeImage() else { return UIImage(cgImage: baseImage) }
  return UIImage(cgImage: composedImage)
}

public func drawYOLOClassifications(on ciImage: CIImage, result: YOLOResult) -> UIImage {
  let context = CIContext(options: nil)
  guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
    return UIImage()
  }
  let width = cgImage.width
  let height = cgImage.height
  let imageSize = CGSize(width: width, height: height)
  UIGraphicsBeginImageContextWithOptions(imageSize, false, 1.0)
  guard let drawContext = UIGraphicsGetCurrentContext() else {
    UIGraphicsEndImageContext()
    return UIImage()
  }
  drawContext.saveGState()
  drawContext.translateBy(x: 0, y: CGFloat(height))
  drawContext.scaleBy(x: 1, y: -1)
  drawContext.draw(cgImage, in: CGRect(origin: .zero, size: imageSize))
  drawContext.restoreGState()
  guard let top5 = result.probs?.top5 else {
    return UIImage(ciImage: ciImage)
  }

  // Calculate line width and font size proportionally to image dimensions
  let lineWidth = max(width, height) / 200
  let fontSize = max(width, height) / 50
  let labelMargin = CGFloat(fontSize / 2)

  for (i, candidate) in top5.enumerated() {
    var colorIndex = 0
    if let index = result.names.firstIndex(of: candidate) {
      colorIndex = index % ultralyticsColors.count
    }
    let color = ultralyticsColors[colorIndex]
    drawContext.setStrokeColor(color.cgColor)
    drawContext.setLineWidth(CGFloat(lineWidth))
    let confidencePercent = round((result.probs?.top5Confs[i] ?? 0) * 1000) / 10
    let labelText = " \(candidate) \(confidencePercent)% "
    let font = UIFont.systemFont(ofSize: CGFloat(fontSize), weight: .semibold)
    let attrs: [NSAttributedString.Key: Any] = [
      .font: font,
      .foregroundColor: UIColor.white,
    ]
    let textSize = labelText.size(withAttributes: attrs)
    let labelWidth = textSize.width + 10
    let labelHeight = CGFloat(textSize.height + 4)
    let labelRect = CGRect(
      x: labelMargin,
      y: labelMargin + (labelHeight + labelMargin) * CGFloat(i),
      width: labelWidth,
      height: labelHeight
    )

    drawContext.setFillColor(color.cgColor)
    drawContext.fill(labelRect)
    let textPoint = CGPoint(
      x: labelRect.origin.x + 5,
      y: labelRect.origin.y + (labelHeight - textSize.height) / 2
    )
    labelText.draw(at: textPoint, withAttributes: attrs)
  }
  let drawnImage = UIGraphicsGetImageFromCurrentImageContext() ?? UIImage()
  UIGraphicsEndImageContext()
  return drawnImage
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

func drawKeypoints(
  keypointsList: [[(x: Float, y: Float)]],
  confsList: [[Float]],
  boundingBoxes: [Box],
  on layer: CALayer,
  imageViewSize: CGSize,
  originalImageSize: CGSize,
  radius: CGFloat = 5,
  confThreshold: Float = 0.25,
  drawSkeleton: Bool = true
) {
  // Scale radius dynamically based on the current view size rather than original image size
  let dynamicRadius = max(imageViewSize.width, imageViewSize.height) / 100
  for (i, keypoints) in keypointsList.enumerated() {
    drawSinglePersonKeypoints(
      keypoints: keypoints, confs: confsList[i], boundingBox: boundingBoxes[i],
      on: layer,
      imageViewSize: imageViewSize,
      originalImageSize: originalImageSize,
      radius: dynamicRadius,
      confThreshold: confThreshold,
      drawSkeleton: drawSkeleton
    )
  }
}

func drawSinglePersonKeypoints(
  keypoints: [(x: Float, y: Float)],
  confs: [Float],
  boundingBox: Box,
  on layer: CALayer,
  imageViewSize: CGSize,
  originalImageSize: CGSize,
  radius: CGFloat,
  confThreshold: Float,
  drawSkeleton: Bool
) {

  let lineWidth = radius * 0.4
  _ = Float(imageViewSize.width / originalImageSize.width)
  _ = Float(imageViewSize.height / originalImageSize.height)

  // Dynamic keypoint count support
  let numKeypoints = keypoints.count
  var points: [(CGPoint, Float)] = Array(repeating: (CGPoint.zero, 0), count: numKeypoints)

  for i in 0..<numKeypoints {
    let x = keypoints[i].x * Float(imageViewSize.width)
    let y = keypoints[i].y * Float(imageViewSize.height)
    let conf = confs[i]

    let point = CGPoint(x: CGFloat(x), y: CGFloat(y))
    let box = boundingBox

    if conf >= confThreshold
      && box.xywhn.contains(CGPoint(x: CGFloat(keypoints[i].x), y: CGFloat(keypoints[i].y)))
    {
      points[i] = (point, conf)

      // Use modulo to cycle through available colors for any number of keypoints
      let colorIndex = i < kptColorIndices.count ? kptColorIndices[i] : i % posePalette.count
      drawCircle(on: layer, at: point, radius: radius, color: colorIndex)
    }
  }

  if drawSkeleton {
    // Only draw skeleton if we have the standard 17 keypoints
    // For other keypoint counts, skeleton connectivity would need model-specific configuration
    if numKeypoints == 17 {
      for (index, bone) in skeleton.enumerated() {
        let (startIdx, endIdx) = (bone[0] - 1, bone[1] - 1)

        guard startIdx < points.count, endIdx < points.count else {
          print("Invalid skeleton indices: \(startIdx), \(endIdx)")
          continue
        }

        let startPoint = points[startIdx].0
        let endPoint = points[endIdx].0
        let startConf = points[startIdx].1
        let endConf = points[endIdx].1

        if startConf >= confThreshold && endConf >= confThreshold {
          let limbColorIndex =
            index < limbColorIndices.count ? limbColorIndices[index] : index % posePalette.count
          drawLine(
            on: layer, from: startPoint, to: endPoint, color: limbColorIndex,
            lineWidth: lineWidth)
        }
      }
    }
  }
}

func drawCircle(on layer: CALayer, at point: CGPoint, radius: CGFloat, color index: Int) {
  let circleLayer = CAShapeLayer()
  circleLayer.path =
    UIBezierPath(
      arcCenter: point,
      radius: radius,
      startAngle: 0,
      endAngle: .pi * 2,
      clockwise: true
    ).cgPath

  let color = posePalette[index].map { $0 / 255.0 }
  circleLayer.fillColor =
    UIColor(red: color[0], green: color[1], blue: color[2], alpha: 1.0).cgColor

  layer.addSublayer(circleLayer)
}

func drawLine(
  on layer: CALayer, from start: CGPoint, to end: CGPoint, color index: Int, lineWidth: CGFloat = 2
) {
  let lineLayer = CAShapeLayer()
  let path = UIBezierPath()
  path.move(to: start)
  path.addLine(to: end)

  lineLayer.path = path.cgPath
  // Ensure minimum line width for visibility
  lineLayer.lineWidth = max(lineWidth, 1.5)

  let color = posePalette[index].map { $0 / 255.0 }
  lineLayer.strokeColor =
    UIColor(red: color[0], green: color[1], blue: color[2], alpha: 1.0).cgColor

  layer.addSublayer(lineLayer)
}

func drawPoseOnCIImage(
  ciImage: CIImage,
  keypointsList: [[(x: Float, y: Float)]],
  confsList: [[Float]],
  boundingBoxes: [Box],
  originalImageSize: CGSize,
  radius: CGFloat = 5,
  confThreshold: Float = 0.25,
  drawSkeleton: Bool = true
) -> UIImage? {
  let context = CIContext(options: nil)

  guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
    return nil
  }

  let renderedWidth = cgImage.width
  let renderedHeight = cgImage.height
  let renderedSize = CGSize(width: renderedWidth, height: renderedHeight)

  // Calculate radius scaled to the rendered image size
  let circleRadius = CGFloat(max(renderedWidth, renderedHeight) / 100)

  UIGraphicsBeginImageContextWithOptions(renderedSize, false, 0.0)
  guard let currentContext = UIGraphicsGetCurrentContext() else {
    return nil
  }

  UIImage(cgImage: cgImage).draw(in: CGRect(origin: .zero, size: renderedSize))

  let rootLayer = CALayer()
  rootLayer.frame = CGRect(origin: .zero, size: renderedSize)

  drawKeypoints(
    keypointsList: keypointsList,
    confsList: confsList,
    boundingBoxes: boundingBoxes,
    on: rootLayer,
    imageViewSize: renderedSize,
    originalImageSize: originalImageSize,
    radius: circleRadius,
    confThreshold: confThreshold,
    drawSkeleton: drawSkeleton
  )

  rootLayer.render(in: currentContext)

  let finalImage = UIGraphicsGetImageFromCurrentImageContext()
  UIGraphicsEndImageContext()

  return finalImage
}

class OBBShapeLayerBundle {
  let shapeLayer = CAShapeLayer()
  let textLayer = CATextLayer()

  init() {
    shapeLayer.strokeColor = UIColor.red.cgColor
    shapeLayer.fillColor = UIColor.clear.cgColor

    textLayer.fontSize = 14
    textLayer.alignmentMode = .left
    textLayer.foregroundColor = UIColor.white.cgColor
    textLayer.cornerRadius = 3
    textLayer.masksToBounds = true
    textLayer.actions = [
      "contents": NSNull(),
      "string": NSNull(),
      "position": NSNull(),
      "bounds": NSNull(),
    ]

    shapeLayer.actions = [
      "strokeColor": NSNull(),
      "fillColor": NSNull(),
      "path": NSNull(),
      "position": NSNull(),
      "bounds": NSNull(),
    ]
  }
}

class OBBRenderer {

  private var layerPool: [OBBShapeLayerBundle] = []
  private var usedLayerCount = 0

  private func getLayerBundle(for parentLayer: CALayer) -> OBBShapeLayerBundle {
    if usedLayerCount < layerPool.count {
      let bundle = layerPool[usedLayerCount]
      bundle.shapeLayer.isHidden = false
      bundle.textLayer.isHidden = false
      usedLayerCount += 1
      return bundle
    } else {
      let newBundle = OBBShapeLayerBundle()
      layerPool.append(newBundle)
      usedLayerCount += 1

      parentLayer.addSublayer(newBundle.shapeLayer)
      parentLayer.addSublayer(newBundle.textLayer)
      return newBundle
    }
  }

  @MainActor func drawObbDetectionsWithReuse(
    obbDetections: [OBBResult],
    on layer: CALayer,
    imageViewSize: CGSize,
    originalImageSize: CGSize,
    lineWidth: CGFloat = 2.0
  ) {
    usedLayerCount = 0

    let scaleX = imageViewSize.width
    let scaleY = imageViewSize.height

    // Calculate line width and font size dynamically based on image dimensions
    let dynamicLineWidth = max(imageViewSize.width, imageViewSize.height) / 200
    let dynamicFontSize = max(imageViewSize.width, imageViewSize.height) / 50

    for detection in obbDetections {
      let bundle = getLayerBundle(for: layer)

      let shapeLayer = bundle.shapeLayer

      let textLayer = bundle.textLayer
      let index = detection.index % ultralyticsColors.count
      let color = ultralyticsColors[index]

      let corners = detection.box.toPolygon()
      let path = UIBezierPath()
      for (i, corner) in corners.enumerated() {
        let px = corner.x * scaleX
        let py = corner.y * scaleY
        if i == 0 {
          path.move(to: CGPoint(x: px, y: py))
        } else {
          path.addLine(to: CGPoint(x: px, y: py))
        }
      }
      path.close()

      shapeLayer.path = path.cgPath
      shapeLayer.strokeColor = color.cgColor
      shapeLayer.fillColor = UIColor.clear.cgColor
      shapeLayer.lineWidth = dynamicLineWidth
      shapeLayer.isHidden = false

      let text = detection.cls + String(format: " %.2f", detection.confidence)
      let font = UIFont.systemFont(ofSize: dynamicFontSize)

      let attributes: [NSAttributedString.Key: Any] = [
        .font: font
      ]
      let textSize = (text as NSString).size(withAttributes: attributes)

      textLayer.font = CGFont(font.fontName as CFString)
      textLayer.fontSize = dynamicFontSize
      textLayer.contentsScale = UIScreen.main.scale
      textLayer.string = text

      textLayer.backgroundColor = color.withAlphaComponent(0.6).cgColor
      textLayer.isHidden = false

      let horizontalPadding: CGFloat = dynamicFontSize / 2
      let verticalPadding: CGFloat = dynamicFontSize / 5

      if let firstCorner = corners.first {
        let px = firstCorner.x * scaleX
        let py = firstCorner.y * scaleY

        textLayer.frame = CGRect(
          x: px,
          y: py - textSize.height - verticalPadding,
          width: textSize.width + horizontalPadding,
          height: textSize.height + verticalPadding
        )
      }
    }

    for i in usedLayerCount..<layerPool.count {
      let bundle = layerPool[i]
      bundle.shapeLayer.isHidden = true
      bundle.textLayer.isHidden = true
    }
  }
}

func drawOBBsOnCIImage(
  ciImage: CIImage,
  obbDetections: [OBBResult],
  targetSize: CGSize? = nil
) -> UIImage? {

  let context = CIContext(options: nil)
  let extent = ciImage.extent
  guard let cgImage = context.createCGImage(ciImage, from: extent) else {
    print("Failed to create CGImage from CIImage")
    return nil
  }

  // Calculate line width and font size proportionally to image dimensions
  let lineWidth: CGFloat = max(extent.width, extent.height) / 200
  let fontSize = max(extent.width, extent.height) / 50
  let outputSize = targetSize ?? CGSize(width: extent.width, height: extent.height)

  UIGraphicsBeginImageContextWithOptions(outputSize, false, 1.0)
  guard let cgContext = UIGraphicsGetCurrentContext() else {
    UIGraphicsEndImageContext()
    return nil
  }

  let baseImage = UIImage(cgImage: cgImage)
  baseImage.draw(in: CGRect(origin: .zero, size: outputSize))
  cgContext.setLineWidth(lineWidth)

  for detection in obbDetections {
    let colorIndex = detection.index % ultralyticsColors.count
    let color = ultralyticsColors[colorIndex]
    cgContext.setStrokeColor(color.cgColor)

    let corners = detection.box.toPolygon()

    cgContext.beginPath()
    for (i, corner) in corners.enumerated() {
      let px = corner.x * outputSize.width
      let py = corner.y * outputSize.height
      if i == 0 {
        cgContext.move(to: CGPoint(x: px, y: py))
      } else {
        cgContext.addLine(to: CGPoint(x: px, y: py))
      }
    }
    cgContext.closePath()
    cgContext.strokePath()

    let labelText = "\(detection.cls) \(String(format: "%.2f", detection.confidence))%"
    let attrs: [NSAttributedString.Key: Any] = [
      .font: UIFont.systemFont(ofSize: fontSize),
      .foregroundColor: UIColor.white,
      .backgroundColor: color.withAlphaComponent(0.7),
    ]
    let textSize = (labelText as NSString).size(withAttributes: attrs)
    let corner0 = corners[0]
    let labelX = corner0.x * outputSize.width
    let labelY = corner0.y * outputSize.height - textSize.height

    (labelText as NSString).draw(
      at: CGPoint(x: labelX, y: labelY),
      withAttributes: attrs
    )
  }

  let drawnImage = UIGraphicsGetImageFromCurrentImageContext()
  UIGraphicsEndImageContext()

  return drawnImage
}

/// Integrated rendering function: Draws both pose keypoints and bounding boxes in a single rendering pass
public func drawYOLOPoseWithBoxes(
  ciImage: CIImage,
  keypointsList: [[(x: Float, y: Float)]],
  confsList: [[Float]],
  boundingBoxes: [Box],
  originalImageSize: CGSize,
  confThreshold: Float = 0.25,
  drawSkeleton: Bool = true
) -> UIImage? {
  // 1. Convert CIImage to CGImage only once
  let context = CIContext(options: nil)
  let extent = ciImage.extent
  guard let cgImage = context.createCGImage(ciImage, from: extent) else {
    print("Failed to create CGImage from CIImage")
    return nil
  }

  let width = CGFloat(cgImage.width)
  let height = CGFloat(cgImage.height)
  let renderedSize = CGSize(width: width, height: height)

  // 2. Calculate drawing sizes
  let circleRadius = max(width, height) / 100
  let lineWidth = max(width, height) / 200
  let fontSize = max(width, height) / 50

  // 3. Create a single rendering context
  UIGraphicsBeginImageContextWithOptions(renderedSize, false, 0.0)
  guard let drawContext = UIGraphicsGetCurrentContext() else {
    UIGraphicsEndImageContext()
    return nil
  }

  // 4. Draw the background image
  drawContext.saveGState()
  drawContext.translateBy(x: 0, y: CGFloat(height))
  drawContext.scaleBy(x: 1, y: -1)
  drawContext.draw(cgImage, in: CGRect(origin: .zero, size: renderedSize))
  drawContext.restoreGState()

  // 5. Draw rounded bounding boxes
  for box in boundingBoxes {
    let colorIndex = box.index % ultralyticsColors.count
    let color = ultralyticsColors[colorIndex]
    drawContext.setStrokeColor(color.cgColor)
    drawContext.setLineWidth(lineWidth)

    // Calculate corner radius (about 5% of box size, minimum 2 pixels)
    let rect = box.xywh
    let cornerRadius = max(min(rect.width, rect.height) * 0.05, 2.0)

    // Draw rounded rectangle
    let boxPath = UIBezierPath(roundedRect: rect, cornerRadius: cornerRadius)
    drawContext.addPath(boxPath.cgPath)
    drawContext.strokePath()

    // Prepare label
    let confidencePercent = Int(box.conf * 100)
    let labelText = "\(box.cls) \(confidencePercent)%"
    let font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
    let attrs: [NSAttributedString.Key: Any] = [
      .font: font,
      .foregroundColor: UIColor.white,
    ]
    let textSize = labelText.size(withAttributes: attrs)
    let labelWidth = textSize.width + 10
    let labelHeight = textSize.height + 4

    // Determine label position
    var labelRect = CGRect(
      x: rect.minX,
      y: rect.minY - labelHeight,
      width: labelWidth,
      height: labelHeight
    )
    if labelRect.minY < 0 {
      labelRect.origin.y = rect.minY
    }

    // Round label background as well
    let labelPath = UIBezierPath(roundedRect: labelRect, cornerRadius: cornerRadius)
    drawContext.setFillColor(color.cgColor)
    drawContext.addPath(labelPath.cgPath)
    drawContext.fillPath()

    // Draw text
    let textPoint = CGPoint(
      x: labelRect.origin.x + 5,
      y: labelRect.origin.y + (labelHeight - textSize.height) / 2
    )
    labelText.draw(at: textPoint, withAttributes: attrs)
  }

  // 6. Create layer for keypoints and skeleton
  let poseLayer = CALayer()
  poseLayer.frame = CGRect(origin: .zero, size: renderedSize)

  // 7. Draw keypoints
  drawKeypoints(
    keypointsList: keypointsList,
    confsList: confsList,
    boundingBoxes: boundingBoxes,
    on: poseLayer,
    imageViewSize: renderedSize,
    originalImageSize: originalImageSize,
    radius: circleRadius,
    confThreshold: confThreshold,
    drawSkeleton: drawSkeleton
  )

  // 8. Composite poseLayer into context
  poseLayer.render(in: drawContext)

  // 9. Generate final image
  let finalImage = UIGraphicsGetImageFromCurrentImageContext()
  UIGraphicsEndImageContext()

  return finalImage
}

/// Integrated rendering function: Draws both segmentation masks and bounding boxes in a single rendering pass
public func drawYOLOSegmentationWithBoxes(
  ciImage: CIImage,
  boxes: [Box],
  maskImage: CGImage?,
  originalImageSize: CGSize
) -> UIImage? {
  // 1. Convert CIImage to CGImage only once
  let context = CIContext(options: nil)
  let extent = ciImage.extent
  guard let cgImage = context.createCGImage(ciImage, from: extent) else {
    print("Failed to create CGImage from CIImage")
    return nil
  }

  let width = CGFloat(cgImage.width)
  let height = CGFloat(cgImage.height)
  let renderedSize = CGSize(width: width, height: height)

  // 2. Calculate drawing sizes
  let lineWidth = max(width, height) / 200
  let fontSize = max(width, height) / 50

  // 3. Create a single rendering context
  UIGraphicsBeginImageContextWithOptions(renderedSize, false, 0.0)
  guard let drawContext = UIGraphicsGetCurrentContext() else {
    UIGraphicsEndImageContext()
    return nil
  }

  // 4. Draw the background image
  drawContext.saveGState()
  drawContext.translateBy(x: 0, y: CGFloat(height))
  drawContext.scaleBy(x: 1, y: -1)
  drawContext.draw(cgImage, in: CGRect(origin: .zero, size: renderedSize))
  drawContext.restoreGState()

  // 5. Overlay semi-transparent mask (if available)
  if let maskImage = maskImage {
    drawContext.saveGState()
    drawContext.setAlpha(0.5)  // Mask transparency

    // Apply the same coordinate transformation as the background image
    // This ensures that the mask is drawn in the same orientation
    drawContext.translateBy(x: 0, y: CGFloat(height))
    drawContext.scaleBy(x: 1, y: -1)

    let baseRect = CGRect(origin: .zero, size: renderedSize)

    // Scale mask if necessary when it has different dimensions from the original image
    _ =
      maskImage.width != Int(width) || maskImage.height != Int(height)
      ? baseRect
      : CGRect(
        x: 0, y: 0, width: CGFloat(maskImage.width), height: CGFloat(maskImage.height))

    // Draw mask image with the correct orientation
    drawContext.draw(maskImage, in: baseRect)
    drawContext.restoreGState()
  }

  // 6. Draw rounded bounding boxes
  for box in boxes {
    let colorIndex = box.index % ultralyticsColors.count
    let color = ultralyticsColors[colorIndex]
    drawContext.setStrokeColor(color.cgColor)
    drawContext.setLineWidth(lineWidth)

    // Calculate corner radius (about 5% of box size, minimum 2 pixels)
    let rect = box.xywh
    let cornerRadius = max(min(rect.width, rect.height) * 0.05, 2.0)

    // Draw rounded rectangle
    let boxPath = UIBezierPath(roundedRect: rect, cornerRadius: cornerRadius)
    drawContext.addPath(boxPath.cgPath)
    drawContext.strokePath()

    // Prepare label
    let confidencePercent = Int(box.conf * 100)
    let labelText = "\(box.cls) \(confidencePercent)%"
    let font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
    let attrs: [NSAttributedString.Key: Any] = [
      .font: font,
      .foregroundColor: UIColor.white,
    ]
    let textSize = labelText.size(withAttributes: attrs)
    let labelWidth = textSize.width + 10
    let labelHeight = textSize.height + 4

    // Determine label position
    var labelRect = CGRect(
      x: rect.minX,
      y: rect.minY - labelHeight,
      width: labelWidth,
      height: labelHeight
    )
    if labelRect.minY < 0 {
      labelRect.origin.y = rect.minY
    }

    // Round label background as well
    let labelPath = UIBezierPath(roundedRect: labelRect, cornerRadius: cornerRadius)
    drawContext.setFillColor(color.cgColor)
    drawContext.addPath(labelPath.cgPath)
    drawContext.fillPath()

    // Draw text
    let textPoint = CGPoint(
      x: labelRect.origin.x + 5,
      y: labelRect.origin.y + (labelHeight - textSize.height) / 2
    )
    labelText.draw(at: textPoint, withAttributes: attrs)
  }

  // 7. Generate final image
  let finalImage = UIGraphicsGetImageFromCurrentImageContext()
  UIGraphicsEndImageContext()

  return finalImage
}
