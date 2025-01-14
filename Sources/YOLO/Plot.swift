import CoreImage
import Foundation
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
