import Foundation
import UIKit
import Vision

public class BasePredictor: Predictor, @unchecked Sendable {
  private(set) var isModelLoaded: Bool = false
  var detector: VNCoreMLModel!
  var visionRequest: VNCoreMLRequest?
  public var labels = [String]()
  var currentBuffer: CVPixelBuffer?
  weak var currentOnResultsListener: ResultsListener?
  weak var currentOnInferenceTimeListener: InferenceTimeListener?
  var inputSize: CGSize!
  var modelInputSize: (width: Int, height: Int) = (0, 0)

  var t0 = 0.0  // inference start
  var t1 = 0.0  // inference dt
  var t2 = 0.0  // inference dt smoothed
  var t3 = CACurrentMediaTime()  // FPS start
  var t4 = 0.0  // FPS dt smoothed
  public var isUpdating: Bool = false

  required init() {

  }

  deinit {
    visionRequest?.cancel()
    visionRequest = nil
  }

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
        if #available(iOS 16.0, *) {
          config.setValue(1, forKey: "experimentalMLE5EngineUsage")
        }

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
        predictor.detector = try VNCoreMLModel(for: mlModel)
        predictor.detector.featureProvider = ThresholdProvider()
        predictor.visionRequest = {
          let request = VNCoreMLRequest(
            model: predictor.detector,
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

  func predict(
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

  var confidenceThreshold = 0.25
  func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
  }

  var iouThreshold = 0.4
  func setIouThreshold(iou: Double) {
    iouThreshold = iou
  }

  var numItemsThreshold = 30
  func setNumItemsThreshold(numItems: Int) {
    numItemsThreshold = numItems
  }

  func processObservations(for request: VNRequest, error: Error?) {

  }

  func predictOnImage(image: CIImage) -> YOLOResult {
    return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
  }

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

    // 入力仕様がImageの場合
    if let imageConstraint = inputDescription.imageConstraint {
      let width = Int(imageConstraint.pixelsWide)
      let height = Int(imageConstraint.pixelsHigh)
      return (width: width, height: height)
    }

    print("an not find input size")
    return (0, 0)
  }
}
