// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import Foundation

/// Represents the different computer vision tasks supported by YOLO models.
public enum YOLOTask: Sendable {
  case detect
  case segment
  case pose
  case obb
  case classify
}
