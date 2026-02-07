// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import YOLOCore

/// SwiftUI Canvas overlay for oriented bounding box results.
public struct OBBOverlay: View {
  public let obbResults: [OBBResult]
  public let viewSize: CGSize

  public init(obbResults: [OBBResult], viewSize: CGSize) {
    self.obbResults = obbResults
    self.viewSize = viewSize
  }

  public var body: some View {
    Canvas { context, size in
      for result in obbResults {
        let color = ultralyticsColors[result.index % ultralyticsColors.count]
        let alpha = Double(max(0, min(1, (result.confidence - 0.2) / 0.8 * 0.9)))

        // Get polygon corners and scale to view
        let polygon = result.box.toPolygon().map { pt in
          CGPoint(x: pt.x * size.width, y: pt.y * size.height)
        }

        guard polygon.count == 4 else { continue }

        // Draw rotated box
        var path = Path()
        path.move(to: polygon[0])
        for i in 1..<polygon.count {
          path.addLine(to: polygon[i])
        }
        path.closeSubpath()
        context.stroke(path, with: .color(color.opacity(alpha)), lineWidth: 3)

        // Draw label at first corner
        let label = String(format: "%@ %.0f%%", result.cls, result.confidence * 100)
        let text = Text(label)
          .font(.system(size: 12, weight: .semibold))
          .foregroundStyle(.white)
        let resolvedText = context.resolve(text)
        let textSize = resolvedText.measure(in: CGSize(width: 300, height: 30))

        // Position label above top-left corner
        let topY = polygon.map(\.y).min() ?? 0
        let leftX = polygon.map(\.x).min() ?? 0
        let labelRect = CGRect(
          x: leftX,
          y: max(0, topY - textSize.height - 2),
          width: textSize.width + 8,
          height: textSize.height + 2
        )
        context.fill(
          RoundedRectangle(cornerRadius: 3).path(in: labelRect),
          with: .color(color.opacity(alpha))
        )
        context.draw(
          resolvedText,
          at: CGPoint(x: labelRect.minX + 4, y: labelRect.minY + 1),
          anchor: .topLeading
        )
      }
    }
  }
}
