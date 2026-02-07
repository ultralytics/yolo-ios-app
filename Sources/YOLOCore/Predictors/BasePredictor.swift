// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import CoreML
import Foundation
import QuartzCore
import Vision

/// Base class for all YOLO model predictors.
///
/// Handles CoreML model loading, initialization, and common inference operations.
/// Task-specific predictors inherit from this class and override prediction methods.
public class BasePredictor: Predictor, @unchecked Sendable {
  private(set) var isModelLoaded: Bool = false

  /// The Vision CoreML model used for inference.
  var detector: VNCoreMLModel?

  /// The Vision request that processes images.
  var visionRequest: VNCoreMLRequest?

  /// The class labels used by the model.
  public private(set) var labels = [String]()

  /// Whether the model requires NMS post-processing (YOLO11 legacy models).
  public private(set) var requiresNMS: Bool = true

  /// Current configuration for thresholds.
  public var configuration: YOLOConfiguration = YOLOConfiguration()

  /// The size of the input image or camera frame.
  var inputSize: CGSize = CGSize(width: 640, height: 640)

  /// The required input dimensions for the model.
  var modelInputSize: (width: Int, height: Int) = (0, 0)

  // Timing
  var t0 = 0.0
  var t1 = 0.0
  var t2 = 0.0
  var t3 = CACurrentMediaTime()
  var t4 = 0.0

  required init() {}

  deinit {
    visionRequest?.cancel()
    visionRequest = nil
  }

  /// Asynchronously creates and initializes a predictor with the specified model.
  public static func create(modelURL: URL) async throws -> Self {
    let predictor = Self.init()

    let ext = modelURL.pathExtension.lowercased()
    let isCompiled = (ext == "mlmodelc")
    let config = MLModelConfiguration()
    config.setValue(1, forKey: "experimentalMLE5EngineUsage")

    let mlModel: MLModel
    if isCompiled {
      mlModel = try MLModel(contentsOf: modelURL, configuration: config)
    } else {
      let compiledUrl = try MLModel.compileModel(at: modelURL)
      mlModel = try MLModel(contentsOf: compiledUrl, configuration: config)
    }

    guard
      let userDefined = mlModel.modelDescription
        .metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]
    else {
      throw PredictorError.modelFileNotFound
    }

    // Extract class labels
    if let labelsData = userDefined["classes"] {
      predictor.labels = labelsData.components(separatedBy: ",")
    } else if let labelsData = userDefined["names"] {
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
          predictor.labels.append(extractedString.replacingOccurrences(of: "'", with: ""))
        }
      }
    } else {
      throw PredictorError.noLabelsFound
    }

    // Detect NMS-free models (Phase 2: YOLO26 support)
    // NMS-free models set "nms" metadata key to "false"; default to requiring NMS for safety
    if let nmsValue = userDefined["nms"] {
      predictor.requiresNMS = (nmsValue.lowercased() != "false")
    } else {
      predictor.requiresNMS = true
    }

    predictor.modelInputSize = predictor.getModelInputSize(for: mlModel)

    let coreMLModel = try VNCoreMLModel(for: mlModel)
    let iou = predictor.requiresNMS ? predictor.configuration.iouThreshold : 1.0
    coreMLModel.featureProvider = ThresholdProvider(
      iouThreshold: iou,
      confidenceThreshold: predictor.configuration.confidenceThreshold
    )
    predictor.detector = coreMLModel

    let request = VNCoreMLRequest(model: coreMLModel)
    request.imageCropAndScaleOption = .scaleFill
    predictor.visionRequest = request

    predictor.isModelLoaded = true
    return predictor
  }

  /// Updates the Vision model's feature provider when configuration changes.
  func updateThresholdProvider() {
    let iou = requiresNMS ? configuration.iouThreshold : 1.0
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iou,
      confidenceThreshold: configuration.confidenceThreshold
    )
  }

  /// Processes a pixel buffer and returns results.
  public func predict(pixelBuffer: CVPixelBuffer) -> YOLOResult {
    inputSize = CGSize(
      width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))

    let handler = VNImageRequestHandler(
      cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
    t0 = CACurrentMediaTime()
    do {
      if let request = visionRequest {
        try handler.perform([request])
      }
    } catch {
      print("Prediction error: \(error)")
    }
    t1 = CACurrentMediaTime() - t0

    // Update smoothed timing
    if t1 < 10.0 {
      t2 = t1 * 0.05 + t2 * 0.95
    }
    t4 = (CACurrentMediaTime() - t3) * 0.05 + t4 * 0.95
    t3 = CACurrentMediaTime()

    return processResults()
  }

  /// Processes a static image and returns results.
  public func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }
    inputSize = CGSize(width: image.extent.width, height: image.extent.height)
    t0 = CACurrentMediaTime()
    do {
      try requestHandler.perform([request])
    } catch {
      print("Prediction error: \(error)")
    }
    t1 = CACurrentMediaTime() - t0
    return processResults()
  }

  /// Override in subclasses to process task-specific results from the Vision request.
  func processResults() -> YOLOResult {
    YOLOResult(orig_shape: inputSize, boxes: [], speed: t1, names: labels)
  }

  func getModelInputSize(for model: MLModel) -> (width: Int, height: Int) {
    guard let inputDescription = model.modelDescription.inputDescriptionsByName.first?.value else {
      return (0, 0)
    }
    if let multiArrayConstraint = inputDescription.multiArrayConstraint {
      let shape = multiArrayConstraint.shape
      if shape.count >= 2 {
        return (width: shape[1].intValue, height: shape[0].intValue)
      }
    }
    if let imageConstraint = inputDescription.imageConstraint {
      return (width: Int(imageConstraint.pixelsWide), height: Int(imageConstraint.pixelsHigh))
    }
    return (0, 0)
  }
}
