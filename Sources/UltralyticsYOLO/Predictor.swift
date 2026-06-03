// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, defining core prediction interfaces.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  Defines the Predictor protocol and listener interfaces shared by every YOLO model implementation. Predictors
//  process static images and camera frames; listeners receive results and timing metrics. The protocol-based design
//  keeps a consistent API across detection, segmentation, semantic segmentation, classification, pose, and OBB
//  tasks. Error types for model loading and inference are also defined here.

import CoreImage
import Vision

/// Callback protocol for receiving YOLO inference results.
///
/// Implementers are notified with processed results when each inference completes.
public protocol ResultsListener: AnyObject {
  /// Called when a new prediction result is available.
  ///
  /// - Parameter result: The processed YOLO model prediction result, containing detections,
  ///   segmentation masks, or other task-specific outputs.
  func on(result: YOLOResult)
}

/// Callback protocol for receiving YOLO inference performance metrics.
///
/// Implementers are notified with timing information so they can monitor inference speed.
public protocol InferenceTimeListener: AnyObject {
  /// Called when inference timing information is available.
  ///
  /// - Parameters:
  ///   - inferenceTime: The time in milliseconds taken to perform the model inference.
  ///   - fpsRate: The calculated frames per second rate based on recent inference times.
  func on(inferenceTime: Double, fpsRate: Double)
}

/// Core protocol for YOLO model predictors.
///
/// Defines the contract every YOLO prediction implementation must fulfill: methods for processing camera frames and
/// static images, plus prediction state. Specialized implementations exist for each task (detection, segmentation,
/// semantic, classification, pose, OBB).
public protocol Predictor {
  /// Processes a camera frame buffer and delivers results via callback.
  ///
  /// - Parameters:
  ///   - sampleBuffer: The camera frame buffer to process.
  ///   - onResultsListener: Optional listener to receive prediction results.
  ///   - onInferenceTime: Optional listener to receive performance metrics.
  func predict(
    sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?,
    onInferenceTime: InferenceTimeListener?)

  /// Processes a static image and returns results synchronously.
  ///
  /// - Parameter image: The CIImage to process.
  /// - Returns: A YOLOResult containing the prediction outputs.
  func predictOnImage(image: CIImage) -> YOLOResult

  /// The class labels used by the model for categorizing detections.
  var labels: [String] { get set }

  /// Flag indicating whether the predictor is currently processing an update.
  var isUpdating: Bool { get set }
}

/// Errors that can occur during YOLO model loading, configuration, or inference.
public enum PredictorError: Error {
  /// The model file could not be found at the specified location.
  case modelFileNotFound
}
