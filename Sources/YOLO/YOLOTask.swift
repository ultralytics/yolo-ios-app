// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, defining the supported inference tasks.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  YOLOTask enumerates the computer-vision tasks YOLO models can perform: object detection, instance segmentation,
//  semantic segmentation, pose estimation, oriented bounding box detection, and image classification. The task
//  selects which predictor implementation the SDK instantiates for a given model.

/// The computer-vision tasks supported by YOLO models.
///
/// Each case maps to a different predictor implementation and output shape. Pick the case that matches the model
/// you are loading.
public enum YOLOTask: Equatable {
  /// Object detection: rectangular bounding boxes with class labels and confidence scores.
  case detect

  /// Instance segmentation: per-object pixel masks alongside bounding boxes.
  case segment

  /// Semantic segmentation: one dense class index per pixel, without separating object instances.
  case semantic

  /// Pose estimation: per-person body keypoints (joints) with confidence scores.
  case pose

  /// Oriented bounding box detection: rotated boxes that fit non-axis-aligned objects more tightly than `detect`.
  case obb

  /// Image classification: top-k class predictions for the full image, with no localization.
  case classify

  /// Parses task names from user/platform strings, defaulting to `.detect` for unknown values.
  ///
  /// Accepts the canonical case names plus common bridge/model aliases such as `"detection"`, `"segmentation"`,
  /// `"classification"`, and `"oriented-bounding-box"`.
  public static func fromString(_ value: String) -> YOLOTask {
    let normalized =
      value
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
      .replacingOccurrences(of: "_", with: "-")
      .replacingOccurrences(of: " ", with: "-")

    switch normalized {
    case "segment", "seg", "segmentation", "instance-segment", "instance-segmentation":
      return .segment
    case "semantic", "semantic-segment", "semantic-segmentation":
      return .semantic
    case "classify", "classification", "class", "cls":
      return .classify
    case "pose", "pose-estimation", "keypoint", "keypoints":
      return .pose
    case "obb", "oriented-bounding-box", "oriented-box", "rotated-box":
      return .obb
    case "detect", "detection", "object-detection":
      return .detect
    default:
      return .detect
    }
  }
}
