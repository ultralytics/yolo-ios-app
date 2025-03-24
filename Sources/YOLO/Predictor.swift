//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
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
protocol ResultsListener: AnyObject {
  func on(result: YOLOResult)
}

protocol InferenceTimeListener: AnyObject {
  func on(inferenceTime: Double, fpsRate: Double)
}
//
//protocol FpsRateListener: AnyObject {
//    func on(fpsRate: Double)
//}

protocol Predictor {
  func predict(
    sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?,
    onInferenceTime: InferenceTimeListener?)
  func predictOnImage(image: CIImage) -> YOLOResult
  var labels: [String] { get set }
  var isUpdating: Bool { get set }
}

enum PredictorError: Error {
  case invalidTask
  case noLabelsFound
  case invalidUrl
  case modelFileNotFound
}
