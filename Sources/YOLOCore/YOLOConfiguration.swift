// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import Foundation

/// Configuration for YOLO model inference thresholds and options.
public struct YOLOConfiguration: Sendable {
  /// Confidence threshold for filtering results (0.0 to 1.0).
  public var confidenceThreshold: Double

  /// IoU threshold for non-maximum suppression (0.0 to 1.0). Only applies to legacy NMS-required models.
  public var iouThreshold: Double

  /// Maximum number of detections to return.
  public var maxDetections: Int

  public init(
    confidenceThreshold: Double = 0.25,
    iouThreshold: Double = 0.45,
    maxDetections: Int = 30
  ) {
    self.confidenceThreshold = confidenceThreshold
    self.iouThreshold = iouThreshold
    self.maxDetections = maxDetections
  }
}
