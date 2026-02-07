// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import YOLOCore

/// SwiftUI Canvas overlay for segmentation mask results.
public struct SegmentationOverlay: View {
  public let boxes: [Box]
  public let masks: Masks?
  public let viewSize: CGSize

  public init(boxes: [Box], masks: Masks?, viewSize: CGSize) {
    self.boxes = boxes
    self.masks = masks
    self.viewSize = viewSize
  }

  public var body: some View {
    ZStack {
      // Draw mask image if available
      if let combinedMask = masks?.combinedMask {
        Image(decorative: combinedMask, scale: 1.0)
          .resizable()
          .scaledToFill()
          .opacity(0.5)
          .allowsHitTesting(false)
      }

      // Draw bounding boxes on top
      DetectionOverlay(boxes: boxes, viewSize: viewSize)
    }
  }
}
