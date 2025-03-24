//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
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
  for box in result.boxes {
    let colorIndex = box.index % ultralyticsColors.count
    let color = ultralyticsColors[colorIndex]
    let lineWidth = CGFloat(width) * 0.01
    drawContext.setStrokeColor(color.cgColor)
    drawContext.setLineWidth(lineWidth)
    let rect = box.xywh
    drawContext.stroke(rect)
    let confidencePercent = Int(box.conf * 100)
    let labelText = "\(box.cls) \(confidencePercent)%"
    let font = UIFont.systemFont(ofSize: CGFloat(width) * 0.03, weight: .semibold)
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
  // 1) protos形状チェック
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

  // 2) 行列A: (N, C) を一括で用意 (オブジェクト数 x マスクチャネル)
  var coeffsArray = [Float](repeating: 0, count: N * maskChannels)
  for i in 0..<N {
    let (_, _, _, coeffsMLArray) = detectedObjects[i]
    let coeffsPtr = coeffsMLArray.dataPointer.assumingMemoryBound(to: Float.self)
    // 行列Aのi行目: coeffsArray[i*C .. i*C + C-1] に書き込み
    for c in 0..<maskChannels {
      coeffsArray[i * maskChannels + c] = coeffsPtr[c]
    }
  }

  // 3) 行列B: (C, HW) は protosPointer をそのまま使用
  //    memory layout が [1, C, H, W] => (C, H, W) => (C, HW) となる。行方向C, 列方向HW
  //    vDSP_mmul は連続メモリを2Dとして扱うだけなのでOK。

  // 4) 行列C(出力): (N, HW) を確保 => combinedMask
  //    フラット形状で N*HW の要素をもつ1次元配列
  var combinedMask = [Float](repeating: 0, count: N * HW)

  // 5) vDSP_mmulで一括演算: (N x C) * (C x HW) => (N x HW)
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

  // 6) スコア順ソート (合成時の描画順を制御)
  //    => (originalIndex, box, classID, score)
  let indexedObjects: [(Int, CGRect, Int, Float)] =
    detectedObjects.enumerated().map { (i, obj) in (i, obj.0, obj.1, obj.2) }
  let sortedObjects = indexedObjects.sorted { $0.3 < $1.3 }  // score昇順

  // 7) RGBAバッファ (160x160)
  var mergedPixels = [UInt8](repeating: 0, count: HW * 4)
  let scaleX = Float(maskWidth) / Float(inputWidth)
  let scaleY = Float(maskHeight) / Float(inputHeight)

  // 8) 個別の確率マップを保持するかどうか
  var probabilityMasks: [[[Float]]]? = nil
  if returnIndividualMasks {
    probabilityMasks = Array(
      repeating: Array(
        repeating: [Float](repeating: 0.0, count: maskWidth),
        count: maskHeight
      ),
      count: N
    )
  }

  // 9) ソート順に従い合成
  for (originalIndex, box, classID, score) in sortedObjects {
    // boundingBoxをマスク座標系に変換
    let minX = Int(Float(box.minX) * scaleX)
    let minY = Int(Float(box.minY) * scaleY)
    let maxX = Int(Float(box.maxX) * scaleX)
    let maxY = Int(Float(box.maxY) * scaleY)

    let boxX1 = max(0, min(minX, maskWidth - 1))
    let boxX2 = max(0, min(maxX, maskWidth - 1))
    let boxY1 = max(0, min(minY, maskHeight - 1))
    let boxY2 = max(0, min(maxY, maskHeight - 1))

    // オブジェクト originalIndex のマスク => combinedMask[originalIndex*HW ..< (originalIndex+1)*HW]
    // フラット配列の先頭
    let startIdx = originalIndex * HW

    // クラス色の取得
    let _colorIndex = classID % ultralyticsColors.count
    let color = ultralyticsColors[_colorIndex].toRGBComponents()!
    let r = UInt8(color.red)
    let g = UInt8(color.green)
    let b = UInt8(color.blue)

    // ピクセルループ: box範囲のみ
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

  // 11) RGBAバッファ -> CGImage
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
  for (i, candidate) in top5.enumerated() {
    var colorIndex = 0
    if let index = result.names.firstIndex(of: candidate) {
      colorIndex = index % ultralyticsColors.count
    }
    let color = ultralyticsColors[colorIndex]
    let lineWidth = CGFloat(width) * 0.01
    drawContext.setStrokeColor(color.cgColor)
    drawContext.setLineWidth(lineWidth)
    let confidencePercent = round(result.probs!.top5Confs[i] * 1000) / 10
    let labelText = " \(candidate) \(confidencePercent)% "
    let fontSize = CGFloat(width) * 0.03
    let font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
    let attrs: [NSAttributedString.Key: Any] = [
      .font: font,
      .foregroundColor: UIColor.white,
    ]
    let textSize = labelText.size(withAttributes: attrs)
    let labelWidth = textSize.width + 10
    let labelHeight = textSize.height + 4
    var labelRect = CGRect(
      x: fontSize,
      y: fontSize + (fontSize * 1.5 * CGFloat(i)),
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
  let _radius = max(originalImageSize.width, originalImageSize.height) / 300
  for (i, keypoints) in keypointsList.enumerated() {
    drawSinglePersonKeypoints(
      keypoints: keypoints, confs: confsList[i], boundingBox: boundingBoxes[i],
      on: layer,
      imageViewSize: imageViewSize,
      originalImageSize: originalImageSize,
      radius: _radius,
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
  //      guard keypoints.count == 17 else {
  //        print("Keypoints array must have 51 elements.")
  //        return
  //      }
  let lineWidth = radius * 0.4
  let scaleXToView = Float(imageViewSize.width / originalImageSize.width)
  let scaleYToView = Float(imageViewSize.height / originalImageSize.height)

  var points: [(CGPoint, Float)] = Array(repeating: (CGPoint.zero, 0), count: 17)

  for i in 0..<17 {
    let x = keypoints[i].x * Float(imageViewSize.width)
    let y = keypoints[i].y * Float(imageViewSize.height)
    let conf = confs[i]

    let point = CGPoint(x: CGFloat(x), y: CGFloat(y))
    let box = boundingBox

    if conf >= confThreshold
      && box.xywhn.contains(CGPoint(x: CGFloat(keypoints[i].x), y: CGFloat(keypoints[i].y)))
    {
      points[i] = (point, conf)

      drawCircle(on: layer, at: point, radius: radius, color: kptColorIndices[i])
    }
  }

  if drawSkeleton {
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
        drawLine(
          on: layer, from: startPoint, to: endPoint, color: limbColorIndices[index],
          lineWidth: lineWidth)
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
  lineLayer.lineWidth = lineWidth

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
  originalImageSize: CGSize,  // 元画像のサイズ(モデルに合わせて)
  radius: CGFloat = 5,
  confThreshold: Float = 0.25,
  drawSkeleton: Bool = true
) -> UIImage? {
  let _radius = max(originalImageSize.width, originalImageSize.height) / 300
  let context = CIContext(options: nil)
  guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
    return nil
  }

  let renderedWidth = cgImage.width
  let renderedHeight = cgImage.height
  let renderedSize = CGSize(width: renderedWidth, height: renderedHeight)

  // 2. UIGraphicsコンテキストを生成し、元画像(CIImage→CGImage)を描画
  UIGraphicsBeginImageContextWithOptions(renderedSize, false, 0.0)
  guard let currentContext = UIGraphicsGetCurrentContext() else {
    return nil
  }

  // ベース画像を描画
  // (CGImageをUIKitのUIImageに変換してから描画)
  UIImage(cgImage: cgImage).draw(in: CGRect(origin: .zero, size: renderedSize))

  // 3. 既存のCALayerを利用した描画ロジックを再利用する
  //    - 一時的にCALayerを作り、そこに既存のdrawKeypointsを呼び出して
  //      最終的に layer.render(in:) でコンテキストへ合成

  let rootLayer = CALayer()
  rootLayer.frame = CGRect(origin: .zero, size: renderedSize)

  // ここでは "imageViewSize" ＝ 実際に描画したい最終サイズ として
  // "renderedSize" を渡す
  drawKeypoints(
    keypointsList: keypointsList,
    confsList: confsList,
    boundingBoxes: boundingBoxes,
    on: rootLayer,
    imageViewSize: renderedSize,
    originalImageSize: originalImageSize,  // 推論時の元画像サイズ
    radius: _radius,
    confThreshold: confThreshold,
    drawSkeleton: drawSkeleton
  )

  // rootLayer をコンテキストに合成
  rootLayer.render(in: currentContext)

  // 4. 描画した結果を UIImage として取得
  let finalImage = UIGraphicsGetImageFromCurrentImageContext()
  UIGraphicsEndImageContext()

  return finalImage
}

/// 使い回し用のレイヤーを保持する構造体例
/// OBB の枠線を描画する CAShapeLayer + オプションの CATextLayer を束ねる
class OBBShapeLayerBundle {
  let shapeLayer = CAShapeLayer()
  let textLayer = CATextLayer()

  init() {
    // 初期設定
    shapeLayer.strokeColor = UIColor.red.cgColor
    shapeLayer.fillColor = UIColor.clear.cgColor

    textLayer.fontSize = 14
    textLayer.alignmentMode = .left
    textLayer.foregroundColor = UIColor.white.cgColor
    textLayer.cornerRadius = 3
    textLayer.masksToBounds = true
    textLayer.actions = [
      "contents": NSNull(),  // 文字列変更(string)の内部でcontents更新の場合
      "string": NSNull(),  // 文字列変更
      "position": NSNull(),  // 位置変更
      "bounds": NSNull(),  // サイズ変更 (frame更新はbounds + position)
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

/// 描画担当クラス (例)
class OBBRenderer {

  /// 使い回し用プール
  private var layerPool: [OBBShapeLayerBundle] = []
  /// このフレームで使用したレイヤー数
  private var usedLayerCount = 0

  /// レイヤーを使い回す時に呼ぶ関数
  private func getLayerBundle(for parentLayer: CALayer) -> OBBShapeLayerBundle {
    if usedLayerCount < layerPool.count {
      // 既存のプールを再利用
      let bundle = layerPool[usedLayerCount]
      bundle.shapeLayer.isHidden = false
      bundle.textLayer.isHidden = false
      usedLayerCount += 1
      return bundle
    } else {
      // プールに無ければ新規生成
      let newBundle = OBBShapeLayerBundle()
      layerPool.append(newBundle)
      usedLayerCount += 1

      // 初回だけ、親に addSublayer しておく
      parentLayer.addSublayer(newBundle.shapeLayer)
      parentLayer.addSublayer(newBundle.textLayer)
      return newBundle
    }
  }

  /// 毎フレーム呼び出して、OBBやテキストを更新する
  /// - Parameters:
  ///   - obbDetections: (OBB, score, cls)配列
  ///   - layer: 親CALayer。例: cameraView.layer など
  ///   - imageViewSize: 表示領域サイズ
  ///   - originalImageSize: 画像やモデルの元サイズ
  ///   - color: 線の色 (実際はクラスごとに変化させたいならループ内で変更)
  ///   - lineWidth: 線の太さ
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

    for detection in obbDetections {
      let bundle = getLayerBundle(for: layer)

      let shapeLayer = bundle.shapeLayer

      let textLayer = bundle.textLayer
      let index = detection.index % ultralyticsColors.count
      let color = ultralyticsColors[index]

      // OBB(四角形)のパスを作成
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

      // shapeLayer設定
      shapeLayer.path = path.cgPath
      shapeLayer.strokeColor = color.cgColor
      shapeLayer.fillColor = UIColor.clear.cgColor
      shapeLayer.lineWidth = lineWidth
      shapeLayer.isHidden = false

      // 文字列 (クラス名 + confidence)
      let text = detection.cls + String(format: " %.2f", detection.confidence)
      // 計算するフォントを用意（UIFont）
      let font = UIFont.systemFont(ofSize: textLayer.fontSize)

      // 文字サイズを事前に計算
      let attributes: [NSAttributedString.Key: Any] = [
        .font: font
      ]
      let textSize = (text as NSString).size(withAttributes: attributes)

      // テキストレイヤーに設定する際は、CATextLayer.font には CGFont をセットする
      // さらに contentsScale を設定しないと文字がぼやけやすい
      textLayer.font = CGFont(font.fontName as CFString)
      textLayer.contentsScale = UIScreen.main.scale
      textLayer.string = text

      // ラベルの背景色など
      textLayer.backgroundColor = color.withAlphaComponent(0.6).cgColor
      textLayer.isHidden = false

      // 余白を少しつけたい場合
      let horizontalPadding: CGFloat = 10
      let verticalPadding: CGFloat = 4

      // corners[0] が左上とは限りませんが、「最初の点の上に表示する」というロジックを踏襲
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

    // 使わなかったレイヤーを非表示にする
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

  let lineWidth: CGFloat = max(extent.width, extent.height) / 500
  let fontSize = max(extent.width, extent.height) / 70
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
