// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage

/// Core protocol for YOLO model predictors.
public protocol Predictor: AnyObject, Sendable {
  /// Processes a pixel buffer and returns results.
  func predict(pixelBuffer: CVPixelBuffer) -> YOLOResult

  /// Processes a static image and returns results.
  func predictOnImage(image: CIImage) -> YOLOResult

  /// The class labels used by the model.
  var labels: [String] { get }

  /// Whether the model requires NMS post-processing (legacy YOLO11 models).
  var requiresNMS: Bool { get }

  /// Current configuration for thresholds.
  var configuration: YOLOConfiguration { get set }
}

/// Errors that can occur during YOLO model prediction.
public enum PredictorError: Error, Sendable {
  case invalidTask
  case noLabelsFound
  case invalidUrl
  case modelFileNotFound
  case modelLoadFailed(String)
  case cameraPermissionDenied
  case cameraSetupFailed
}
