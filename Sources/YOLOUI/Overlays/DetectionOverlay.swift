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

/// SwiftUI Canvas overlay for bounding box detection results.
public struct DetectionOverlay: View {
  public let boxes: [Box]
  public let viewSize: CGSize

  public init(boxes: [Box], viewSize: CGSize) {
    self.boxes = boxes
    self.viewSize = viewSize
  }

  public var body: some View {
    Canvas { context, size in
      for box in boxes {
        let color = ultralyticsColors[box.index % ultralyticsColors.count]
        let alpha = Double(max(0, min(1, (box.conf - 0.2) / 0.8 * 0.9)))

        // Convert normalized coordinates to view coordinates
        let rect = CGRect(
          x: CGFloat(box.xywhn.minX) * size.width,
          y: CGFloat(box.xywhn.minY) * size.height,
          width: CGFloat(box.xywhn.width) * size.width,
          height: CGFloat(box.xywhn.height) * size.height
        )

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
