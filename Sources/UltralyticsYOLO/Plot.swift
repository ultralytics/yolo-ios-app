// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing visualization utilities.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Plot module renders YOLO inference results. It draws bounding boxes, segmentation masks, pose keypoints,
//  classification labels, and oriented bounding boxes onto images, with per-class color management and support for
//  both static-image and real-time scenarios.

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

func makeRGBAImage(
  from pixels: Data,
  width: Int,
  height: Int,
  shouldInterpolate: Bool = true
) -> CGImage? {
  guard let provider = CGDataProvider(data: pixels as CFData) else { return nil }
  return CGImage(
    width: width,
    height: height,
    bitsPerComponent: 8,
    bitsPerPixel: 32,
    bytesPerRow: width * 4,
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
    provider: provider,
    decode: nil,
    shouldInterpolate: shouldInterpolate,
    intent: .defaultIntent)
}

/// Executes `body` inside a bitmap graphics context rendered at pixel scale.
///
/// Flips the y-axis so drawing matches UIKit's top-left origin and draws the source image as the background, then
/// invokes `body` with the context and pixel size for callers to add their overlays.
private func renderWithBackground(
  _ ciImage: CIImage,
  targetSize: CGSize? = nil,
  _ body: (CGContext, CGSize) -> Void
) -> UIImage? {
  let context = CIContext(options: nil)
  let extent = ciImage.extent
  guard let cgImage = context.createCGImage(ciImage, from: extent) else {
    YOLOLog.error("Failed to create CGImage from CIImage")
    return nil
  }
  let size = targetSize ?? CGSize(width: cgImage.width, height: cgImage.height)
  UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
  defer { UIGraphicsEndImageContext() }
  guard let drawContext = UIGraphicsGetCurrentContext() else { return nil }

  drawContext.saveGState()
  drawContext.translateBy(x: 0, y: size.height)
  drawContext.scaleBy(x: 1, y: -1)
  drawContext.draw(cgImage, in: CGRect(origin: .zero, size: size))
  drawContext.restoreGState()

  body(drawContext, size)
  return UIGraphicsGetImageFromCurrentImageContext()
}

private func drawDetectionLabel(
  _ labelText: String,
  in ctx: CGContext,
  fontSize: CGFloat,
  color: UIColor,
  alpha: CGFloat,
  anchor: CGPoint,
  cornerRadius: CGFloat
) {
  let labelRect = DetectionLabelStyle.frame(for: labelText, fontSize: fontSize, anchor: anchor)
  ctx.setFillColor(color.withAlphaComponent(alpha).cgColor)
  let labelPath = UIBezierPath(
    roundedRect: labelRect,
    cornerRadius: min(DetectionLabelStyle.cornerRadius, cornerRadius)
  )
  ctx.addPath(labelPath.cgPath)
  ctx.fillPath()

  let textSize = labelText.size(withAttributes: DetectionLabelStyle.attributes(fontSize: fontSize))
  let textPoint = CGPoint(
    x: labelRect.origin.x + DetectionLabelStyle.horizontalPadding / 2,
    y: labelRect.origin.y + (labelRect.height - textSize.height) / 2
  )
  labelText.draw(
    at: textPoint,
    withAttributes: DetectionLabelStyle.attributes(fontSize: fontSize, alpha: alpha)
  )
}

/// Stroked label + box drawing shared by the detection/pose/segmentation renderers.
///
/// - Parameter rounded: pass `true` for the "rounded corner" style used by pose/segment
///   overlays, and `false` for the straight-corner style used by raw detections.
private func drawBoxLabel(
  _ box: Box,
  in ctx: CGContext,
  imageSize: CGSize,
  rounded: Bool
) {
  let color = ultralyticsColors[box.index % ultralyticsColors.count]
  ctx.setStrokeColor(color.cgColor)
  let lineWidth = max(imageSize.width, imageSize.height) / 200
  ctx.setLineWidth(lineWidth)

  let rect = box.xywh
  if rounded {
    let cornerRadius = max(min(rect.width, rect.height) * 0.05, 2.0)
    let boxPath = UIBezierPath(roundedRect: rect, cornerRadius: cornerRadius)
    ctx.addPath(boxPath.cgPath)
    ctx.strokePath()
  } else {
    ctx.stroke(rect)
  }

  let fontSize = max(imageSize.width, imageSize.height) / 50
  let labelText = DetectionLabelStyle.text(className: box.cls, confidence: CGFloat(box.conf))
  drawDetectionLabel(
    labelText,
    in: ctx,
    fontSize: fontSize,
    color: color,
    alpha: 1,
    anchor: rect.origin,
    cornerRadius: max(min(rect.width, rect.height) * 0.05, 2.0)
  )
}

public func drawYOLODetections(on ciImage: CIImage, result: YOLOResult) -> UIImage {
  renderWithBackground(ciImage) { ctx, size in
    for box in result.boxes {
      drawBoxLabel(box, in: ctx, imageSize: size, rounded: false)
    }
  } ?? UIImage()
}

func generateCombinedMaskImage(
  detectedObjects: [(CGRect, Int, Float, [Float])],
  protos: MLMultiArray,  // shape: [1, C, H, W]
  inputWidth: Int,
  inputHeight: Int,
  threshold: Float = 0.0,
  cropRect: CGRect? = nil,
  returnIndividualMasks: Bool = true
) -> (CGImage?, [[[Float]]]?)? {
  let maskHeight = protos.shape[2].intValue  // e.g. 160
  let maskWidth = protos.shape[3].intValue  // e.g. 160
  let maskChannels = protos.shape[1].intValue  // e.g. 32
  guard
    protos.shape.count == 4,
    protos.shape[0].intValue == 1,
    maskHeight > 0,
    maskWidth > 0,
    maskChannels > 0
  else {
    YOLOLog.error("Invalid prototype mask shape: \(protos.shape)")
    return nil
  }

  let protosPointer = protos.dataPointer.assumingMemoryBound(to: Float.self)
  let HW = maskHeight * maskWidth
  let N = detectedObjects.count

  // No detections: nothing to composite. Return early so the empty-buffer unsafe-pointer paths below
  // (vDSP_mmul, the row copies) are never reached with a nil base address.
  guard N > 0 else { return (nil, returnIndividualMasks ? [] : nil) }

  // 2) Prepare matrix A: (N, C) at once (number of objects x mask channels)
  var coeffsArray = [Float](repeating: 0, count: N * maskChannels)
  for i in 0..<N {
    let coeffs = detectedObjects[i].3
    // Row i of matrix A: write to coeffsArray[i*C .. i*C + C-1]
    for c in 0..<min(maskChannels, coeffs.count) {
      coeffsArray[i * maskChannels + c] = coeffs[c]
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

  // 7) Display geometry. Mask math stays at prototype resolution, but the visible composite is painted at
  // model-input resolution so the app does not upscale a 160x160 image into blocky masks.
  let scaleX = Float(maskWidth) / Float(inputWidth)
  let scaleY = Float(maskHeight) / Float(inputHeight)
  let maskBounds = CGRect(x: 0, y: 0, width: maskWidth, height: maskHeight)
  let outputRect = (cropRect ?? maskBounds).intersection(maskBounds).integral
  let outputX = Int(outputRect.minX)
  let outputY = Int(outputRect.minY)
  let outputWidth = Int(outputRect.width)
  let outputHeight = Int(outputRect.height)
  guard outputWidth > 0, outputHeight > 0 else { return nil }
  let targetWidth = max(
    1, Int((CGFloat(outputWidth) / CGFloat(maskWidth) * CGFloat(inputWidth)).rounded()))
  let targetHeight = max(
    1, Int((CGFloat(outputHeight) / CGFloat(maskHeight) * CGFloat(inputHeight)).rounded()))
  let displayScaleX = CGFloat(targetWidth) / CGFloat(outputWidth)
  let displayScaleY = CGFloat(targetHeight) / CGFloat(outputHeight)
  var mergedPixels = [UInt32](repeating: 0, count: targetWidth * targetHeight)

  // Match Ultralytics process_mask(): crop each mask to its detection box before thresholding
  // or returning per-instance data.
  var maskBoxes: [(x1: Int, y1: Int, x2: Int, y2: Int)] = []
  maskBoxes.reserveCapacity(N)
  for (box, _, _, _) in detectedObjects {
    let x1 = Int((Float(box.minX) * scaleX).rounded())
    let y1 = Int((Float(box.minY) * scaleY).rounded())
    let x2 = Int((Float(box.maxX) * scaleX).rounded())
    let y2 = Int((Float(box.maxY) * scaleY).rounded())
    maskBoxes.append(
      (
        x1: max(0, min(x1, maskWidth)),
        y1: max(0, min(y1, maskHeight)),
        x2: max(0, min(x2, maskWidth)),
        y2: max(0, min(y2, maskHeight))
      ))
  }

  if returnIndividualMasks {
    for i in 0..<N {
      let box = maskBoxes[i]
      let startIdx = i * HW
      for y in 0..<maskHeight {
        let rowStart = startIdx + y * maskWidth
        let insideY = y >= box.y1 && y < box.y2
        for x in 0..<maskWidth where !insideY || x < box.x1 || x >= box.x2 {
          combinedMask[rowStart + x] = 0
        }
      }
    }
  }

  // 8) Per-instance probability maps are built later (section 10) with bulk row copies.
  var probabilityMasks: [[[Float]]]? = nil
  var scaledMask = [Float]()

  // 9) Compose according to sort order. Scale each instance's Float logits before thresholding so edges are
  // high-resolution binary masks, not blurred RGBA pixels.
  for (originalIndex, _, classID, _) in sortedObjects {
    let maskBox = maskBoxes[originalIndex]
    guard maskBox.x1 < maskBox.x2, maskBox.y1 < maskBox.y2 else { continue }

    let startIdx = originalIndex * HW
    let visibleBox = CGRect(
      x: maskBox.x1, y: maskBox.y1, width: maskBox.x2 - maskBox.x1,
      height: maskBox.y2 - maskBox.y1
    ).intersection(outputRect).integral
    let sourceX = Int(visibleBox.minX)
    let sourceY = Int(visibleBox.minY)
    let sourceWidth = Int(visibleBox.width)
    let sourceHeight = Int(visibleBox.height)
    guard sourceWidth > 0, sourceHeight > 0 else { continue }

    let targetX1 = max(0, Int(((visibleBox.minX - outputRect.minX) * displayScaleX).rounded(.down)))
    let targetY1 = max(0, Int(((visibleBox.minY - outputRect.minY) * displayScaleY).rounded(.down)))
    let targetX2 = min(
      targetWidth, Int(((visibleBox.maxX - outputRect.minX) * displayScaleX).rounded(.up)))
    let targetY2 = min(
      targetHeight, Int(((visibleBox.maxY - outputRect.minY) * displayScaleY).rounded(.up)))
    let targetBoxWidth = targetX2 - targetX1
    let targetBoxHeight = targetY2 - targetY1
    guard targetBoxWidth > 0, targetBoxHeight > 0 else { continue }

    // Get class color
    let _colorIndex = classID % ultralyticsColors.count
    guard let color = ultralyticsColors[_colorIndex].toRGBComponents() else {
      continue
    }
    let r = UInt8(color.red)
    let g = UInt8(color.green)
    let b = UInt8(color.blue)
    let colorWord = UInt32(r) | UInt32(g) << 8 | UInt32(b) << 16 | 0xFF00_0000

    let scaledCount = targetBoxWidth * targetBoxHeight
    if scaledMask.count < scaledCount {
      scaledMask = [Float](repeating: 0, count: scaledCount)
    }
    let scaleError = combinedMask.withUnsafeBufferPointer { sourceBuffer in
      scaledMask.withUnsafeMutableBufferPointer { targetBuffer in
        var source = vImage_Buffer(
          data: UnsafeMutableRawPointer(
            mutating: sourceBuffer.baseAddress! + startIdx + sourceY * maskWidth + sourceX),
          height: vImagePixelCount(sourceHeight),
          width: vImagePixelCount(sourceWidth),
          rowBytes: maskWidth * MemoryLayout<Float>.stride)
        var target = vImage_Buffer(
          data: targetBuffer.baseAddress!,
          height: vImagePixelCount(targetBoxHeight),
          width: vImagePixelCount(targetBoxWidth),
          rowBytes: targetBoxWidth * MemoryLayout<Float>.stride)
        return vImageScale_PlanarF(&source, &target, nil, vImage_Flags(kvImageNoFlags))
      }
    }
    guard scaleError == kvImageNoError else { continue }

    for y in 0..<targetBoxHeight {
      let sourceRow = y * targetBoxWidth
      let targetRow = (targetY1 + y) * targetWidth + targetX1
      for x in 0..<targetBoxWidth where scaledMask[sourceRow + x] > threshold {
        mergedPixels[targetRow + x] = colorWord
      }
    }
  }

  // 10) Per-instance probability maps. Copy each output row in one bulk operation from the contiguous
  //     `combinedMask` buffer instead of writing element-by-element through `[[[Float]]]` subscripts. The
  //     nested-array form forces per-element ARC/COW/bounds-check overhead; bulk row copies are ~13x faster
  //     for typical sizes (e.g. 30 instances × 160×160) while producing bit-identical values.
  if returnIndividualMasks {
    probabilityMasks = combinedMask.withUnsafeBufferPointer { buf -> [[[Float]]] in
      guard let base = buf.baseAddress else { return [] }
      var masksArray = [[[Float]]]()
      masksArray.reserveCapacity(N)
      for i in 0..<N {
        let startIdx = i * HW
        var rows = [[Float]]()
        rows.reserveCapacity(outputHeight)
        for y in 0..<outputHeight {
          let rowStart = startIdx + (outputY + y) * maskWidth + outputX
          rows.append(Array(UnsafeBufferPointer(start: base + rowStart, count: outputWidth)))
        }
        masksArray.append(rows)
      }
      return masksArray
    }
  }

  let pixelData = mergedPixels.withUnsafeBufferPointer { Data(buffer: $0) }
  return (
    makeRGBAImage(from: pixelData, width: targetWidth, height: targetHeight), probabilityMasks
  )
}

public func drawYOLOClassifications(on ciImage: CIImage, result: YOLOResult) -> UIImage {
  guard let top5 = result.probs?.top5 else { return UIImage(ciImage: ciImage) }

  return renderWithBackground(ciImage) { ctx, size in
    let fontSize = max(size.width, size.height) / 50
    let labelMargin = fontSize / 2

    for (i, candidate) in top5.enumerated() {
      let colorIndex = (result.names.firstIndex(of: candidate) ?? 0) % ultralyticsColors.count
      let color = ultralyticsColors[colorIndex]
      let labelText = DetectionLabelStyle.text(
        className: candidate,
        confidence: CGFloat(result.probs?.top5Confs[i] ?? 0)
      )
      let textSize = DetectionLabelStyle.size(for: labelText, fontSize: fontSize)
      let labelRect = CGRect(
        x: labelMargin,
        y: labelMargin + (textSize.height + labelMargin) * CGFloat(i),
        width: textSize.width,
        height: textSize.height)

      ctx.setFillColor(color.cgColor)
      let labelPath = UIBezierPath(
        roundedRect: labelRect,
        cornerRadius: DetectionLabelStyle.cornerRadius
      )
      ctx.addPath(labelPath.cgPath)
      ctx.fillPath()
      let textPoint = CGPoint(
        x: labelRect.origin.x + DetectionLabelStyle.horizontalPadding / 2,
        y: labelRect.origin.y)
      labelText.draw(
        at: textPoint, withAttributes: DetectionLabelStyle.attributes(fontSize: fontSize))
    }
  } ?? UIImage()
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
  originalImageSize: CGSize? = nil,
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
  radius: CGFloat,
  confThreshold: Float,
  drawSkeleton: Bool
) {
  let lineWidth = radius * 0.4

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
          YOLOLog.warning("Invalid skeleton indices: \(startIdx), \(endIdx)")
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

func drawOBBsOnCIImage(
  ciImage: CIImage,
  obbDetections: [OBBResult],
  targetSize: CGSize? = nil
) -> UIImage? {
  renderWithBackground(ciImage, targetSize: targetSize) { ctx, size in
    let lineWidth: CGFloat = max(size.width, size.height) / 200
    let fontSize = max(size.width, size.height) / 50
    ctx.setLineWidth(lineWidth)

    for detection in obbDetections {
      let color = ultralyticsColors[detection.index % ultralyticsColors.count]
      ctx.setStrokeColor(color.cgColor)

      // Compute polygon in pixel space to avoid aspect-ratio distortion.
      let corners = detection.box.toPolygon(imageSize: size)
      ctx.beginPath()
      for (i, corner) in corners.enumerated() {
        i == 0 ? ctx.move(to: corner) : ctx.addLine(to: corner)
      }
      ctx.closePath()
      ctx.strokePath()

      if let first = corners.first {
        drawDetectionLabel(
          DetectionLabelStyle.text(
            className: detection.cls,
            confidence: CGFloat(detection.confidence)
          ),
          in: ctx,
          fontSize: fontSize,
          color: color,
          alpha: 1,
          anchor: first,
          cornerRadius: DetectionLabelStyle.cornerRadius
        )
      }
    }
  }
}

/// Renders pose keypoints, skeleton, and rounded bounding boxes onto the source image.
public func drawYOLOPoseWithBoxes(
  ciImage: CIImage,
  keypointsList: [[(x: Float, y: Float)]],
  confsList: [[Float]],
  boundingBoxes: [Box],
  confThreshold: Float = 0.25,
  drawSkeleton: Bool = true,
  originalImageSize: CGSize? = nil
) -> UIImage? {
  renderWithBackground(ciImage) { ctx, size in
    for box in boundingBoxes {
      drawBoxLabel(box, in: ctx, imageSize: size, rounded: true)
    }
    let poseLayer = CALayer()
    poseLayer.frame = CGRect(origin: .zero, size: size)
    drawKeypoints(
      keypointsList: keypointsList,
      confsList: confsList,
      boundingBoxes: boundingBoxes,
      on: poseLayer,
      imageViewSize: size,
      originalImageSize: originalImageSize ?? size,
      confThreshold: confThreshold,
      drawSkeleton: drawSkeleton)
    poseLayer.render(in: ctx)
  }
}

/// Renders segmentation masks plus rounded bounding boxes onto the source image.
public func drawYOLOSegmentationWithBoxes(
  ciImage: CIImage,
  boxes: [Box],
  maskImage: CGImage?
) -> UIImage? {
  renderWithBackground(ciImage) { ctx, size in
    if let maskImage = maskImage {
      ctx.saveGState()
      ctx.setAlpha(0.5)
      // Flip to match the background orientation applied by renderWithBackground.
      ctx.translateBy(x: 0, y: size.height)
      ctx.scaleBy(x: 1, y: -1)
      ctx.draw(maskImage, in: CGRect(origin: .zero, size: size))
      ctx.restoreGState()
    }
    for box in boxes {
      drawBoxLabel(box, in: ctx, imageSize: size, rounded: true)
    }
  }
}

/// Renders a semantic segmentation color map onto the source image.
public func drawYOLOSemanticSegmentation(
  ciImage: CIImage,
  semanticMask: CGImage?
) -> UIImage? {
  renderWithBackground(ciImage) { ctx, size in
    if let semanticMask = semanticMask {
      ctx.saveGState()
      ctx.setAlpha(0.5)
      ctx.translateBy(x: 0, y: size.height)
      ctx.scaleBy(x: 1, y: -1)
      ctx.draw(semanticMask, in: CGRect(origin: .zero, size: size))
      ctx.restoreGState()
    }
  }
}
