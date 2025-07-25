// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, defining core prediction interfaces.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Predictor protocol and related interfaces define the contract for all YOLO model prediction
//  implementations. This includes methods for processing images and camera frames, as well as
//  listener protocols for receiving prediction results and performance metrics. The protocol-based
//  design enables a consistent API across different model types (detection, segmentation, classification)
//  while allowing for specialized implementations. Error types related to prediction processes
//  are also defined here, providing standardized error handling throughout the application.

import CoreImage
import Vision

/// Protocol for receiving YOLO model prediction results.
///
/// This protocol defines a callback mechanism for receiving results from YOLO model inference.
/// Implementers receive notifications with processed results when model inference is complete.
protocol ResultsListener: AnyObject {
  /// Called when a new prediction result is available.
  ///
  /// - Parameter result: The processed YOLO model prediction result, containing detections,
  ///   segmentation masks, or other task-specific outputs.
  func on(result: YOLOResult)
}

/// Protocol for receiving model inference performance metrics.
///
/// This protocol defines a callback mechanism for monitoring the performance of YOLO model inference.
/// Implementers receive notifications with timing information to track inference speed.
protocol InferenceTimeListener: AnyObject {
  /// Called when inference timing information is available.
  ///
  /// - Parameters:
  ///   - inferenceTime: The time in seconds taken to perform the model inference.
  ///   - fpsRate: The calculated frames per second rate based on recent inference times.
  func on(inferenceTime: Double, fpsRate: Double)
}
//
//protocol FpsRateListener: AnyObject {
//    func on(fpsRate: Double)
//}

/// Core protocol for YOLO model predictors.
///
/// This protocol defines the contract that all YOLO model prediction implementations must fulfill.
/// It provides methods for processing both camera frames and static images, and managing prediction state.
/// Specialized implementations exist for different model types (detection, segmentation, etc.).
protocol Predictor: AnyObject {
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

/// Errors that can occur during YOLO model prediction.
///
/// This enumeration defines the different types of errors that may be encountered
/// during model loading, configuration, and inference operations.
enum PredictorError: Error {
  /// The requested task type is not supported or invalid.
  case invalidTask

  /// No class labels were found for the model.
  case noLabelsFound

  /// The provided URL for model or resource loading is invalid.
  case invalidUrl

  /// The model file could not be found at the specified location.
  case modelFileNotFound
}
