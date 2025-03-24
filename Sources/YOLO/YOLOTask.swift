//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Ultralytics YOLO Package, defining the supported inference tasks.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOTask enum defines the different types of computer vision tasks that the YOLO models can perform.
//  Each task represents a distinct type of machine learning capability, from basic object detection to
//  more advanced tasks like instance segmentation, pose estimation, oriented bounding box detection,
//  and image classification. This enum is used throughout the application to configure the model loading
//  and inference pipeline for the specific task selected by the user.

import Foundation

/// Represents the different computer vision tasks supported by YOLO models.
public enum YOLOTask {
  case detect
  case segment
  case pose
  case obb
  case classify
}
