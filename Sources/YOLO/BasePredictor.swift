// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO SDK, providing the base infrastructure for model prediction.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  BasePredictor is the foundation for all task-specific predictors. It loads Core ML models asynchronously on a
//  background thread, extracts class labels, measures inference timing, and exposes confidence and IoU thresholds.
//  Task-specific subclasses (detection, segmentation, semantic segmentation, classification, pose, OBB) override the
//  prediction methods to handle their own output formats.

import Foundation
import UIKit
import Vision

/// Base class for all YOLO model predictors, handling common model loading and inference logic.
///
/// Manages Core ML model loading, initialization, and shared inference operations. Specialized subclasses (detection,
/// segmentation, classification, pose, OBB) override the prediction methods to handle task-specific processing.
///
/// - Note: Marked `@unchecked Sendable` to support concurrent use across threads.
/// - Important: Subclasses must override `processObservations` and `predictOnImage`.
public class BasePredictor: Predictor, @unchecked Sendable {
  /// Flag indicating if the model has been successfully loaded and is ready for inference.
  private(set) var isModelLoaded: Bool = false

  /// The Vision Core ML model used for inference operations.
  var detector: VNCoreMLModel?

  /// The Vision request that processes images using the Core ML model.
  var visionRequest: VNCoreMLRequest?

  /// Vision preprocessing mode for this predictor. Localization tasks use Ultralytics LetterBox-style aspect-fit
  /// preprocessing; classification overrides this to center crop.
  var imageCropAndScaleOption: VNImageCropAndScaleOption { .scaleFit }

  /// The class labels used by the model for categorizing detections.
  public var labels = [String]()

  /// Whether the model requires NMS post-processing (false for YOLO26 nms-free models).
  public private(set) var requiresNMS: Bool = true

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
  var t4 = 1.0  // FPS dt smoothed (non-zero to avoid infinity on first frame)

  /// EMA weight for new samples in smoothed inference/FPS measurements.
  private static let emaAlpha = 0.05

  /// Maximum plausible per-frame delta (seconds); outliers above this are ignored.
  private static let maxValidDt = 10.0

  /// Flag indicating whether the predictor is currently processing an update.
  public var isUpdating: Bool = false

  /// Required initializer for the factory pattern used by `create`. Subclasses may override to add initialization.
  required init() {
    // Intentionally left empty
  }

  /// Releases the Vision request on deinit so its completion handler can no longer retain `self`.
  deinit {
    visionRequest = nil
  }

  /// Dispatches `create` to the concrete predictor type for the given task, centralizing the task → predictor mapping.
  public static func create(
    for task: YOLOTask,
    modelURL: URL,
    isRealTime: Bool = false,
    completion: @escaping @Sendable (Result<BasePredictor, Error>) -> Void
  ) {
    switch task {
    case .classify:
      Classifier.create(unwrappedModelURL: modelURL, isRealTime: isRealTime, completion: completion)
    case .segment:
      Segmenter.create(unwrappedModelURL: modelURL, isRealTime: isRealTime, completion: completion)
    case .semantic:
      SemanticSegmenter.create(
        unwrappedModelURL: modelURL, isRealTime: isRealTime, completion: completion)
    case .pose:
      PoseEstimator.create(
        unwrappedModelURL: modelURL, isRealTime: isRealTime, completion: completion)
    case .obb:
      ObbDetector.create(
        unwrappedModelURL: modelURL, isRealTime: isRealTime, completion: completion)
    case .detect:
      ObjectDetector.create(
        unwrappedModelURL: modelURL, isRealTime: isRealTime, completion: completion)
    }
  }

  /// Asynchronously creates and initializes a predictor with the specified model.
  ///
  /// Loads the Core ML model on a background thread, then invokes the completion handler on the main thread with the
  /// initialized predictor or an error.
  ///
  /// - Parameters:
  ///   - unwrappedModelURL: The URL of the Core ML model file to load.
  ///   - isRealTime: Pass `true` when the predictor will be driven by a camera feed.
  ///   - completion: Callback that receives the initialized predictor or an error.
  public static func create(
    unwrappedModelURL: URL,
    isRealTime: Bool = false,
    completion: @escaping @Sendable (Result<BasePredictor, Error>) -> Void
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
        // Pin inference to the Apple Neural Engine (plus CPU fallback), excluding the GPU. In a real-time camera app
        // the GPU is busy compositing the preview and overlays; letting CoreML schedule conv/decode work on the GPU
        // (.all) risks contention and frame-time jitter. .cpuAndNeuralEngine keeps the conv backbone on the Apple
        // Neural Engine — the engine these models already prefer — and was verified on a physical iPhone.
        config.computeUnits = .cpuAndNeuralEngine

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

        // Detect NMS-free models (YOLO26 support)
        if let nmsValue = userDefined["nms"] {
          predictor.requiresNMS = (nmsValue.lowercased() != "false")
        }

        // (3) Store model input size
        predictor.modelInputSize = predictor.getModelInputSize(for: mlModel)

        // (4) Create VNCoreMLModel, VNCoreMLRequest, etc.
        let coreMLModel = try VNCoreMLModel(for: mlModel)
        let iou = predictor.requiresNMS ? predictor.iouThreshold : 1.0
        coreMLModel.featureProvider = ThresholdProvider(
          iouThreshold: iou, confidenceThreshold: predictor.confidenceThreshold)
        predictor.detector = coreMLModel
        predictor.visionRequest = {
          let request = VNCoreMLRequest(model: coreMLModel)
          request.imageCropAndScaleOption = predictor.imageCropAndScaleOption
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

  /// Runs inference on a camera frame buffer and delivers results via callbacks.
  ///
  /// Called repeatedly with frames from a camera feed; runs the Vision request and notifies listeners with the
  /// detection results and performance metrics.
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

      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor, so in most cases Vision needs to rotate it for the
      // model to work as expected.
      let imageOrientation: CGImagePropertyOrientation = .up

      // Invoke a VNRequestHandler with that image
      let handler = VNImageRequestHandler(
        cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
      t0 = CACurrentMediaTime()  // inference start
      do {
        if let request = visionRequest {
          try handler.perform([request])
          processObservations(for: request, nil)
        } else {
          isUpdating = false
        }
      } catch {
        YOLOLog.error("Vision request failed: \(error)")
        isUpdating = false
      }

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
    let iou = requiresNMS ? iouThreshold : 1.0
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iou, confidenceThreshold: confidenceThreshold)
  }

  /// The IoU (Intersection over Union) threshold for non-maximum suppression (default: 0.7).
  var iouThreshold = 0.7

  /// Sets the IoU threshold for non-maximum suppression.
  ///
  /// - Parameter iou: The new IoU threshold value (0.0 to 1.0).
  func setIouThreshold(iou: Double) {
    iouThreshold = iou
    let effectiveIou = requiresNMS ? iouThreshold : 1.0
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: effectiveIou, confidenceThreshold: confidenceThreshold)
  }

  /// The maximum number of detections to return in results (default: 30).
  var numItemsThreshold = 30

  /// Sets the maximum number of detection items to include in results.
  ///
  /// - Parameter numItems: The maximum number of items to include.
  func setNumItemsThreshold(numItems: Int) {
    numItemsThreshold = numItems
  }

  /// Processes Vision framework observations from model inference.
  ///
  /// Invoked when Vision completes a request with the model's outputs. Subclasses must override to parse task-specific
  /// outputs (detection boxes, segmentation masks, etc.).
  ///
  /// - Parameters:
  ///   - request: The completed Vision request containing model outputs.
  ///   - error: Any error that occurred during the Vision request.
  func processObservations(for request: VNRequest, _ error: Error?) {
    // Base implementation is empty - must be overridden by subclasses
  }

  /// Runs synchronous inference on a static image. Subclasses must override to implement task-specific processing.
  ///
  /// - Parameter image: The CIImage to process.
  /// - Returns: A YOLOResult containing the prediction outputs.
  public func predictOnImage(image: CIImage) -> YOLOResult {
    // Base implementation returns an empty result - must be overridden by subclasses
    return .empty
  }

  /// Extracts the required input dimensions from the model description, used to properly size images before inference.
  ///
  /// - Parameter model: The Core ML model to analyze.
  /// - Returns: A tuple containing the width and height in pixels required by the model.
  func getModelInputSize(for model: MLModel) -> (width: Int, height: Int) {
    guard let inputDescription = model.modelDescription.inputDescriptionsByName.first?.value else {
      YOLOLog.warning("Model has no input description")
      return (0, 0)
    }

    if let multiArrayConstraint = inputDescription.multiArrayConstraint {
      let shape = multiArrayConstraint.shape
      if shape.count >= 2 {
        let height = shape[shape.count - 2].intValue
        let width = shape[shape.count - 1].intValue
        return (width: width, height: height)
      }
    }

    if let imageConstraint = inputDescription.imageConstraint {
      let width = Int(imageConstraint.pixelsWide)
      let height = Int(imageConstraint.pixelsHigh)
      return (width: width, height: height)
    }

    YOLOLog.warning("Could not determine model input size")
    return (0, 0)
  }

  private func letterboxTransform(
    inputSize: CGSize,
    modelInputSize: (width: Int, height: Int)
  ) -> (gain: CGFloat, padX: CGFloat, padY: CGFloat)? {
    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)
    let inputWidth = inputSize.width
    let inputHeight = inputSize.height
    guard modelWidth > 0, modelHeight > 0, inputWidth > 0, inputHeight > 0 else { return nil }

    let gain = min(modelHeight / inputHeight, modelWidth / inputWidth)
    guard gain > 0 else { return nil }
    let resizedWidth = (inputWidth * gain).rounded()
    let resizedHeight = (inputHeight * gain).rounded()
    let padX = ((modelWidth - resizedWidth) / 2 - 0.1).rounded()
    let padY = ((modelHeight - resizedHeight) / 2 - 0.1).rounded()
    return (gain, padX, padY)
  }

  private func letterboxTransform() -> (gain: CGFloat, padX: CGFloat, padY: CGFloat)? {
    letterboxTransform(inputSize: inputSize, modelInputSize: modelInputSize)
  }

  func inputRect(fromModelRect rect: CGRect) -> CGRect {
    guard let transform = letterboxTransform() else { return .zero }
    let x1 = (rect.minX - transform.padX) / transform.gain
    let y1 = (rect.minY - transform.padY) / transform.gain
    let x2 = (rect.maxX - transform.padX) / transform.gain
    let y2 = (rect.maxY - transform.padY) / transform.gain

    let minX = min(max(min(x1, x2), 0), inputSize.width)
    let minY = min(max(min(y1, y2), 0), inputSize.height)
    let maxX = min(max(max(x1, x2), 0), inputSize.width)
    let maxY = min(max(max(y1, y2), 0), inputSize.height)
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
  }

  func normalizedRect(fromInputRect rect: CGRect) -> CGRect {
    guard inputSize.width > 0, inputSize.height > 0 else { return .zero }
    return CGRect(
      x: rect.minX / inputSize.width,
      y: rect.minY / inputSize.height,
      width: rect.width / inputSize.width,
      height: rect.height / inputSize.height)
  }

  func inputPoint(fromModelPoint point: CGPoint) -> CGPoint {
    guard let transform = letterboxTransform() else { return .zero }
    let x = (point.x - transform.padX) / transform.gain
    let y = (point.y - transform.padY) / transform.gain
    return CGPoint(
      x: min(max(x, 0), inputSize.width),
      y: min(max(y, 0), inputSize.height))
  }

  func normalizedPoint(fromInputPoint point: CGPoint) -> CGPoint {
    guard inputSize.width > 0, inputSize.height > 0 else { return .zero }
    return CGPoint(x: point.x / inputSize.width, y: point.y / inputSize.height)
  }

  func inputOBB(fromModelOBB box: OBB) -> OBB {
    guard let transform = letterboxTransform(), inputSize.width > 0, inputSize.height > 0 else {
      return OBB(cx: 0, cy: 0, w: 0, h: 0, angle: 0)
    }
    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)
    let centerX = (CGFloat(box.cx) * modelWidth - transform.padX) / transform.gain
    let centerY = (CGFloat(box.cy) * modelHeight - transform.padY) / transform.gain
    let width = CGFloat(box.w) * modelWidth / transform.gain
    let height = CGFloat(box.h) * modelHeight / transform.gain
    return OBB(
      cx: Float(centerX / inputSize.width),
      cy: Float(centerY / inputSize.height),
      w: Float(width / inputSize.width),
      h: Float(height / inputSize.height),
      angle: box.angle)
  }

  func inputMaskCropRect(
    maskWidth: Int,
    maskHeight: Int,
    inputSize: CGSize,
    modelInputSize: (width: Int, height: Int)
  ) -> CGRect? {
    guard
      let transform = letterboxTransform(inputSize: inputSize, modelInputSize: modelInputSize)
    else { return nil }

    let maskWidth = CGFloat(maskWidth)
    let maskHeight = CGFloat(maskHeight)
    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)
    let padWidth = modelWidth - (inputSize.width * transform.gain).rounded()
    let padHeight = modelHeight - (inputSize.height * transform.gain).rounded()
    let left = ((padWidth / 2 - 0.1).rounded() / modelWidth * maskWidth).rounded()
    let top = ((padHeight / 2 - 0.1).rounded() / modelHeight * maskHeight).rounded()
    let right =
      maskWidth - ((padWidth / 2 + 0.1).rounded() / modelWidth * maskWidth).rounded()
    let bottom =
      maskHeight - ((padHeight / 2 + 0.1).rounded() / modelHeight * maskHeight).rounded()
    let cropRect = CGRect(x: left, y: top, width: right - left, height: bottom - top)
      .intersection(CGRect(x: 0, y: 0, width: maskWidth, height: maskHeight))
    guard cropRect.width > 0, cropRect.height > 0 else { return nil }
    if cropRect == CGRect(x: 0, y: 0, width: maskWidth, height: maskHeight) { return nil }
    return cropRect
  }

  /// Updates the smoothed inference time and FPS, then notifies the timing listener.
  ///
  /// Call this once per processed frame after `t1` is set. Uses an EMA with `emaAlpha` weight on new samples and skips
  /// obvious outliers above `maxValidDt`.
  func updateTime(notify: Bool = true) {
    let alpha = Self.emaAlpha
    let now = CACurrentMediaTime()
    self.t1 = now - self.t0
    if self.t1 < Self.maxValidDt {  // valid dt
      self.t2 = self.t1 * alpha + self.t2 * (1 - alpha)  // smoothed inference time
    }
    self.t4 = (now - self.t3) * alpha + self.t4 * (1 - alpha)  // smoothed FPS dt
    self.t3 = now

    if notify {
      self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)
    }
  }
}
