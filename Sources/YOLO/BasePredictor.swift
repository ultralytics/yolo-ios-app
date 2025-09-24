// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing the base infrastructure for model prediction.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The BasePredictor class is the foundation for all task-specific predictors in the YOLO framework.
//  It manages the loading and initialization of CoreML models, handling common operations such as
//  model loading, class label extraction, and inference timing. The class provides an asynchronous
//  model loading mechanism that runs on background threads and includes support for configuring
//  model parameters like confidence thresholds and IoU thresholds. Specific task implementations
//  (detection, segmentation, classification, etc.) inherit from this base class and override
//  the prediction-specific methods.

import Foundation
import UIKit
import Vision

/// Base class for all YOLO model predictors, handling common model loading and inference logic.
///
/// The BasePredictor serves as the foundation for all task-specific YOLO model predictors.
/// It manages CoreML model loading, initialization, and common inference operations.
/// Specialized predictors (for detection, segmentation, etc.) inherit from this class
/// and override the prediction-specific methods to handle task-specific processing.
///
/// - Note: This class is marked as `@unchecked Sendable` to support concurrent operations.
/// - Important: Task-specific implementations must override the `processObservations` and
///   `predictOnImage` methods to provide proper functionality.
public class BasePredictor: Predictor, @unchecked Sendable {
  /// Flag indicating if the model has been successfully loaded and is ready for inference.
  private(set) var isModelLoaded: Bool = false

  /// The Vision CoreML model used for inference operations.
  var detector: VNCoreMLModel?

  /// The Vision request that processes images using the CoreML model.
  var visionRequest: VNCoreMLRequest?

  /// The class labels used by the model for categorizing detections.
  public var labels = [String]()

  /// The current pixel buffer being processed (used for camera frame processing).
  var currentBuffer: CVPixelBuffer?

  /// The current listener to receive prediction results.
  weak var currentOnResultsListener: ResultsListener?

  /// The current listener to receive inference timing information.
  weak var currentOnInferenceTimeListener: InferenceTimeListener?

  /// The size of the input image or camera frame.
  var inputSize: CGSize = CGSize(width: 640, height: 640)

  /// The required input dimensions for the model (width and height in pixels).
  var modelInputSize: (width: Int, height: Int) = (0, 0)

  /// Timestamp for the start of inference (used for performance measurement).
  var t0 = 0.0  // inference start

  /// Duration of a single inference operation.
  var t1 = 0.0  // inference dt

  /// Smoothed inference duration (averaged over recent operations).
  var t2 = 0.0  // inference dt smoothed

  /// Timestamp for FPS calculation start (used for performance measurement).
  var t3 = CACurrentMediaTime()  // FPS start

  /// Smoothed frames per second measurement (averaged over recent frames).
  var t4 = 0.0  // FPS dt smoothed

  /// Flag indicating whether the predictor is currently processing an update.
  public var isUpdating: Bool = false

  /// Required initializer for creating predictor instances.
  ///
  /// This empty initializer is required for the factory pattern used in the `create` method.
  /// Subclasses may override this to perform additional initialization.
  required init() {
    // Intentionally left empty
  }

  /// Performs cleanup when the predictor is deallocated.
  ///
  /// Cancels any pending vision requests and releases references to avoid memory leaks.
  deinit {
    visionRequest?.cancel()
    visionRequest = nil
  }

  /// Factory method to asynchronously create and initialize a predictor with the specified model.
  ///
  /// This method loads the CoreML model in a background thread and sets up the prediction
  /// infrastructure. The completion handler is called on the main thread with either a
  /// successfully initialized predictor or an error.
  ///
  /// - Parameters:
  ///   - unwrappedModelURL: The URL of the CoreML model file to load.
  ///   - isRealTime: Flag indicating if the predictor will be used for real-time processing (camera feed).
  ///   - completion: Callback that receives the initialized predictor or an error.
  /// - Note: Model loading happens on a background thread to avoid blocking the main thread.
  public static func create(
    unwrappedModelURL: URL,
    isRealTime: Bool = false,
    completion: @escaping (Result<BasePredictor, Error>) -> Void
  ) {
    // Create an instance (synchronously, cheap)
    let predictor = Self.init()

    // Kick off the expensive loading on a background thread
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        // (1) Load the MLModel
        let ext = unwrappedModelURL.pathExtension.lowercased()
        let isCompiled = (ext == "mlmodelc")
        let config = MLModelConfiguration()

        let mlModel: MLModel
        if isCompiled {
          mlModel = try MLModel(contentsOf: unwrappedModelURL, configuration: config)
        } else {
          let compiledUrl = try MLModel.compileModel(at: unwrappedModelURL)
          mlModel = try MLModel(contentsOf: compiledUrl, configuration: config)
        }

        guard
          let userDefined = mlModel.modelDescription
            .metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]
        else {
          throw PredictorError.modelFileNotFound
        }

        // (2) Extract class labels
        if let labelsData = userDefined["classes"] {
          predictor.labels = labelsData.components(separatedBy: ",")
        } else if let labelsData = userDefined["names"] {
          // Parse JSON/dictionary-ish format
          let cleanedInput =
            labelsData
            .replacingOccurrences(of: "{", with: "")
            .replacingOccurrences(of: "}", with: "")
            .replacingOccurrences(of: " ", with: "")
          let keyValuePairs = cleanedInput.components(separatedBy: ",")
          for pair in keyValuePairs {
            let components = pair.components(separatedBy: ":")
            if components.count >= 2 {
              let extractedString = components[1].trimmingCharacters(in: .whitespaces)
              let cleanedString = extractedString.replacingOccurrences(of: "'", with: "")
              predictor.labels.append(cleanedString)
            }
          }
        } else {
          throw NSError(
            domain: "BasePredictor", code: -1,
            userInfo: [
              NSLocalizedDescriptionKey: "Invalid metadata format"
            ])
        }

        // (3) Store model input size
        predictor.modelInputSize = predictor.getModelInputSize(for: mlModel)

        // (4) Create VNCoreMLModel, VNCoreMLRequest, etc.
        let coreMLModel = try VNCoreMLModel(for: mlModel)
        coreMLModel.featureProvider = ThresholdProvider()
        predictor.detector = coreMLModel
        predictor.visionRequest = {
          let request = VNCoreMLRequest(
            model: coreMLModel,
            completionHandler: {
              [weak predictor] request, error in
              guard let predictor = predictor else {
                // The predictor was deallocated — do nothing
                return
              }
              if isRealTime {
                predictor.processObservations(for: request, error: error)
              }
            })
          request.imageCropAndScaleOption = .scaleFill
          return request
        }()

        // Once done, mark it loaded
        predictor.isModelLoaded = true

        // Finally, call the completion on the main thread
        DispatchQueue.main.async {
          completion(.success(predictor))
        }
      } catch {
        // If anything goes wrong, call completion with the error
        DispatchQueue.main.async {
          completion(.failure(error))
        }
      }
    }
  }

  /// Processes a camera frame buffer and delivers results via callbacks.
  ///
  /// This method takes a camera sample buffer, performs inference using the Vision framework,
  /// and notifies listeners with the results and performance metrics. It's designed to be
  /// called repeatedly with frames from a camera feed.
  ///
  /// - Parameters:
  ///   - sampleBuffer: The camera frame buffer to process.
  ///   - onResultsListener: Optional listener to receive prediction results.
  ///   - onInferenceTime: Optional listener to receive performance metrics.
  public func predict(
    sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?,
    onInferenceTime: InferenceTimeListener?
  ) {
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer
      inputSize = CGSize(
        width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
      currentOnResultsListener = onResultsListener
      currentOnInferenceTimeListener = onInferenceTime
      //            currentOnFpsRateListener = onFpsRate

      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
      let imageOrientation: CGImagePropertyOrientation = .up

      // Invoke a VNRequestHandler with that image
      let handler = VNImageRequestHandler(
        cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
      t0 = CACurrentMediaTime()  // inference start
      do {
        if visionRequest != nil {
          try handler.perform([visionRequest!])
        }
      } catch {
        print(error)
      }
      t1 = CACurrentMediaTime() - t0  // inference dt

      currentBuffer = nil
    }
  }

  /// The confidence threshold for filtering detection results (default: 0.25).
  ///
  /// Only detections with confidence scores above this threshold will be included in results.
  var confidenceThreshold = 0.25

  /// Sets the confidence threshold for filtering results.
  ///
  /// - Parameter confidence: The new confidence threshold value (0.0 to 1.0).
  func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
  }

  /// The IoU (Intersection over Union) threshold for non-maximum suppression (default: 0.4).
  ///
  /// Used to filter overlapping detections during non-maximum suppression.
  var iouThreshold = 0.4

  /// Sets the IoU threshold for non-maximum suppression.
  ///
  /// - Parameter iou: The new IoU threshold value (0.0 to 1.0).
  func setIouThreshold(iou: Double) {
    iouThreshold = iou
  }

  /// The maximum number of detections to return in results (default: 30).
  ///
  /// Limits the number of detection items in the final results to prevent overwhelming processing.
  var numItemsThreshold = 30

  /// Sets the maximum number of detection items to include in results.
  ///
  /// - Parameter numItems: The maximum number of items to include.
  func setNumItemsThreshold(numItems: Int) {
    numItemsThreshold = numItems
  }

  /// Processes Vision framework observations from model inference.
  ///
  /// This method is called when Vision completes a request with the model's outputs.
  /// Subclasses must override this method to implement task-specific processing of the
  /// model's output features (e.g., parsing detection boxes, segmentation masks, etc.).
  ///
  /// - Parameters:
  ///   - request: The completed Vision request containing model outputs.
  ///   - error: Any error that occurred during the Vision request.
  func processObservations(for request: VNRequest, error: Error?) {
    // Base implementation is empty - must be overridden by subclasses
  }

  /// Processes a static image and returns results synchronously.
  ///
  /// This method performs model inference on a static image and returns the results.
  /// Subclasses must override this method to implement task-specific processing.
  ///
  /// - Parameter image: The CIImage to process.
  /// - Returns: A YOLOResult containing the prediction outputs.
  public func predictOnImage(image: CIImage) -> YOLOResult {
    // Base implementation returns an empty result - must be overridden by subclasses
    return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
  }

  /// Extracts the required input dimensions from the model description.
  ///
  /// This utility method determines the expected input size for the CoreML model
  /// by examining its input description, which is essential for properly sizing
  /// and formatting images before inference.
  ///
  /// - Parameter model: The CoreML model to analyze.
  /// - Returns: A tuple containing the width and height in pixels required by the model.
  func getModelInputSize(for model: MLModel) -> (width: Int, height: Int) {
    guard let inputDescription = model.modelDescription.inputDescriptionsByName.first?.value else {
      print("can not find input description")
      return (0, 0)
    }

    if let multiArrayConstraint = inputDescription.multiArrayConstraint {
      let shape = multiArrayConstraint.shape
      if shape.count >= 2 {
        let height = shape[0].intValue
        let width = shape[1].intValue
        return (width: width, height: height)
      }
    }

    if let imageConstraint = inputDescription.imageConstraint {
      let width = Int(imageConstraint.pixelsWide)
      let height = Int(imageConstraint.pixelsHigh)
      return (width: width, height: height)
    }

    print("an not find input size")
    return (0, 0)
  }
}
