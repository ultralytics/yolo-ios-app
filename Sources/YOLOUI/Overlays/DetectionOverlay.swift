// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import YOLOCore

/// Ultralytics color palette for detection visualizations.
let ultralyticsColors: [Color] = [
  Color(red: 4 / 255, green: 42 / 255, blue: 255 / 255),
  Color(red: 11 / 255, green: 219 / 255, blue: 235 / 255),
  Color(red: 243 / 255, green: 243 / 255, blue: 243 / 255),
  Color(red: 0 / 255, green: 223 / 255, blue: 183 / 255),
  Color(red: 17 / 255, green: 31 / 255, blue: 104 / 255),
  Color(red: 255 / 255, green: 111 / 255, blue: 221 / 255),
  Color(red: 255 / 255, green: 68 / 255, blue: 79 / 255),
  Color(red: 204 / 255, green: 237 / 255, blue: 0 / 255),
  Color(red: 0 / 255, green: 243 / 255, blue: 68 / 255),
  Color(red: 189 / 255, green: 0 / 255, blue: 255 / 255),
  Color(red: 0 / 255, green: 180 / 255, blue: 255 / 255),
  Color(red: 221 / 255, green: 0 / 255, blue: 186 / 255),
  Color(red: 0 / 255, green: 255 / 255, blue: 255 / 255),
  Color(red: 38 / 255, green: 192 / 255, blue: 0 / 255),
  Color(red: 1 / 255, green: 255 / 255, blue: 179 / 255),
  Color(red: 125 / 255, green: 36 / 255, blue: 255 / 255),
  Color(red: 123 / 255, green: 0 / 255, blue: 104 / 255),
  Color(red: 255 / 255, green: 27 / 255, blue: 108 / 255),
  Color(red: 252 / 255, green: 109 / 255, blue: 47 / 255),
  Color(red: 162 / 255, green: 255 / 255, blue: 11 / 255),
]

/// Maps normalized coordinates (0–1, relative to camera frame) to view coordinates,
/// matching `AVCaptureVideoPreviewLayer.resizeAspectFill` behavior.
struct AspectFillTransform {
  let scaleX: CGFloat
  let scaleY: CGFloat
  let offsetX: CGFloat
  let offsetY: CGFloat

  init(frameSize: CGSize, viewSize: CGSize) {
    let scale = max(viewSize.width / frameSize.width, viewSize.height / frameSize.height)
    let displayW = frameSize.width * scale
    let displayH = frameSize.height * scale
    self.scaleX = displayW
    self.scaleY = displayH
    self.offsetX = (displayW - viewSize.width) / 2
    self.offsetY = (displayH - viewSize.height) / 2
  }

  func point(nx: CGFloat, ny: CGFloat) -> CGPoint {
    CGPoint(x: nx * scaleX - offsetX, y: ny * scaleY - offsetY)
  }

  func rect(_ nr: CGRect) -> CGRect {
    CGRect(
      x: nr.minX * scaleX - offsetX,
      y: nr.minY * scaleY - offsetY,
      width: nr.width * scaleX,
      height: nr.height * scaleY)
  }
}

/// SwiftUI Canvas overlay for bounding box detection results.
public struct DetectionOverlay: View {
  public let boxes: [Box]
  public let frameSize: CGSize
  public let viewSize: CGSize

  public init(boxes: [Box], frameSize: CGSize, viewSize: CGSize) {
    self.boxes = boxes
    self.frameSize = frameSize
    self.viewSize = viewSize
  }

  public var body: some View {
    Canvas { context, size in
      let transform = AspectFillTransform(frameSize: frameSize, viewSize: size)

      for box in boxes {
        let color = ultralyticsColors[box.index % ultralyticsColors.count]
        let alpha = Double(max(0, min(1, (box.conf - 0.2) / 0.8 * 0.9)))

        // Convert normalized coordinates to view coordinates with aspect-fill mapping
        let rect = transform.rect(box.xywhn)

        // Draw box
        let path = RoundedRectangle(cornerRadius: 4).path(in: rect)
        context.stroke(path, with: .color(color.opacity(alpha)), lineWidth: 3)

        // Draw label
        let label = String(format: "%@ %.0f%%", box.cls, box.conf * 100)
        var text = context.resolve(
          Text(label).font(.system(size: 12, weight: .semibold)))
        text.shading = .color(.white)
        let textSize = text.measure(in: CGSize(width: 300, height: 30))

        let labelRect = CGRect(
          x: rect.minX,
          y: max(0, rect.minY - textSize.height - 2),
          width: textSize.width + 8,
          height: textSize.height + 2
        )
        context.fill(
          RoundedRectangle(cornerRadius: 3).path(in: labelRect),
          with: .color(color.opacity(alpha))
        )
        context.draw(
          text,
          at: CGPoint(x: labelRect.minX + 4, y: labelRect.minY + 1),
          anchor: .topLeading
        )
      }
    }
  }
}
