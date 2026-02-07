// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import YOLOCore

/// Pose skeleton connectivity (17-keypoint COCO format).
private let skeleton: [(Int, Int)] = [
  (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
  (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
  (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
  (1, 3), (2, 4), (3, 5), (4, 6),
]

/// Color indices for each skeleton limb.
private let limbColorIndices = [
  9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16,
]

/// Color indices for each keypoint.
private let kptColorIndices = [
  16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9,
]

/// Pose color palette (RGB).
private let posePalette: [(Int, Int, Int)] = [
  (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0), (255, 153, 255),
  (153, 204, 255), (255, 102, 255), (255, 51, 255), (102, 178, 255), (51, 153, 255),
  (255, 153, 153), (255, 102, 102), (255, 51, 0), (255, 0, 0), (255, 0, 51),
  (255, 0, 102), (128, 128, 255), (0, 153, 255), (255, 204, 153), (255, 255, 102),
]

/// SwiftUI Canvas overlay for pose estimation results.
public struct PoseOverlay: View {
  public let boxes: [Box]
  public let keypointsList: [Keypoints]
  public let viewSize: CGSize

  public init(boxes: [Box], keypointsList: [Keypoints], viewSize: CGSize) {
    self.boxes = boxes
    self.keypointsList = keypointsList
    self.viewSize = viewSize
  }

  public var body: some View {
    ZStack {
      // Draw bounding boxes
      DetectionOverlay(boxes: boxes, viewSize: viewSize)

      // Draw skeletons
      Canvas { context, size in
        for keypoints in keypointsList {
          // Draw limbs
          for (i, bone) in skeleton.enumerated() {
            let (from, to) = bone
            guard from < keypoints.xyn.count, to < keypoints.xyn.count else { continue }
            guard keypoints.conf[from] > 0.25, keypoints.conf[to] > 0.25 else { continue }

            let p1 = CGPoint(
              x: CGFloat(keypoints.xyn[from].x) * size.width,
              y: CGFloat(keypoints.xyn[from].y) * size.height
            )
            let p2 = CGPoint(
              x: CGFloat(keypoints.xyn[to].x) * size.width,
              y: CGFloat(keypoints.xyn[to].y) * size.height
            )

            let colorIdx = limbColorIndices[i % limbColorIndices.count]
            let (r, g, b) = posePalette[colorIdx % posePalette.count]
            let color = Color(
              red: Double(r) / 255, green: Double(g) / 255, blue: Double(b) / 255)

            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(color), lineWidth: 2)
          }

          // Draw keypoints
          for (i, kpt) in keypoints.xyn.enumerated() {
            guard i < keypoints.conf.count, keypoints.conf[i] > 0.25 else { continue }

            let center = CGPoint(
              x: CGFloat(kpt.x) * size.width,
              y: CGFloat(kpt.y) * size.height
            )

            let colorIdx = kptColorIndices[i % kptColorIndices.count]
            let (r, g, b) = posePalette[colorIdx % posePalette.count]
            let color = Color(
              red: Double(r) / 255, green: Double(g) / 255, blue: Double(b) / 255)

            let radius: CGFloat = 4
            let rect = CGRect(
              x: center.x - radius, y: center.y - radius,
              width: radius * 2, height: radius * 2
            )
            context.fill(Circle().path(in: rect), with: .color(color))
          }
        }
      }
    }
  }
}
