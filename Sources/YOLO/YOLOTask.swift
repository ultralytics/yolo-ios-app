// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, defining the supported inference tasks.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOTask enum defines the different types of computer vision tasks that the YOLO models can perform.
//  Each task represents a distinct type of machine learning capability, from basic object detection to
//  more advanced tasks like instance segmentation, pose estimation, oriented bounding box detection,
//  and image classification. This enum is used throughout the application to configure the model loading
//  and inference pipeline for the specific task selected by the user.

/// Represents the different computer vision tasks supported by YOLO models.
///
/// This enumeration defines the various computer vision tasks that can be performed
/// by YOLO models, each requiring different model architectures and processing pipelines.
/// The task type determines how the model processes input images and what kind of
/// results it produces.
public enum YOLOTask {
  /// Standard object detection task that identifies and localizes objects with bounding boxes.
  ///
  /// Detection models produce rectangular bounding boxes around detected objects
  /// along with class labels and confidence scores.
  case detect

  /// Instance segmentation task that creates pixel-level masks for detected objects.
  ///
  /// Segmentation models produce precise object outlines (masks) for each detected object,
  /// providing more detailed boundaries than rectangular bounding boxes.
  case segment

  /// Human pose estimation task that identifies key points of human figures.
  ///
  /// Pose estimation models detect human figures and identify the positions of key body
  /// joints and points, useful for tracking human movements and poses.
  case pose

  /// Oriented bounding box detection for objects at various angles.
  ///
  /// OBB models detect objects with rotated bounding boxes, providing better fitting
  /// boundaries for objects that are not aligned with the image axes.
  case obb

  /// Image classification task that identifies the primary subject of an image.
  ///
  /// Classification models predict what an image contains without localizing objects,
  /// returning class labels and confidence scores for the entire image.
  case classify
}
