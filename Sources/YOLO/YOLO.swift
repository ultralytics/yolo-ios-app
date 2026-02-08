// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing the main entry point for using YOLO models.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLO class serves as the primary interface for loading and using YOLO machine learning models.
//  It supports a variety of input formats including UIImage, CIImage, CGImage, and resource files.
//  The class handles model loading, format conversion, and inference execution, offering a simple yet
//  powerful API through Swift's callable object pattern. Users can load models from local bundles or
//  file paths and perform inference with a single function call syntax, making integration into iOS
//  applications straightforward.

import Foundation
import SwiftUI
import UIKit

/// The primary interface for working with YOLO models, supporting multiple input types and inference methods.
public class YOLO: @unchecked Sendable {
  var predictor: Predictor!

  private var pendingNumItems: Int?
  private var pendingConfidence: Double?
  private var pendingIou: Double?

  /// Initialize YOLO with remote URL for automatic download and caching
  public init(url: URL, task: YOLOTask, completion: @escaping (Result<YOLO, Error>) -> Void) {
    let downloader = YOLOModelDownloader()

    downloader.download(from: url, task: task) { result in
      switch result {
      case .success(let modelPath):
        self.loadModel(from: modelPath, task: task, completion: completion)
      case .failure(let error):
        completion(.failure(error))
      }
    }
  }

  public init(
    _ modelPathOrName: String, task: YOLOTask, completion: ((Result<YOLO, Error>) -> Void)? = nil
  ) {
    var modelURL: URL?

    let lowercasedPath = modelPathOrName.lowercased()
    let fileManager = FileManager.default

    if lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage") {
      let possibleURL = URL(fileURLWithPath: modelPathOrName)
      if fileManager.fileExists(atPath: possibleURL.path) {
        modelURL = possibleURL
      }
    } else {
      if let compiledURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc")
      {
        modelURL = compiledURL
      } else if let packageURL = Bundle.main.url(
        forResource: modelPathOrName, withExtension: "mlpackage")
      {
        modelURL = packageURL
      }
    }

    guard let unwrappedModelURL = modelURL else {
      completion?(.failure(PredictorError.modelFileNotFound))
      return
    }

    loadModel(from: unwrappedModelURL, task: task, completion: completion)
  }

  /// Load model from URL with task-specific predictor creation
  private func loadModel(
    from modelURL: URL, task: YOLOTask, completion: ((Result<YOLO, Error>) -> Void)?
  ) {
    let handleResult: (Result<BasePredictor, Error>) -> Void = { result in
      switch result {
      case .success(let predictor):
        self.predictor = predictor
        self.pendingNumItems.map { predictor.setNumItemsThreshold(numItems: $0) }
        self.pendingConfidence.map { predictor.setConfidenceThreshold(confidence: $0) }
        self.pendingIou.map { predictor.setIouThreshold(iou: $0) }
        self.pendingNumItems = nil
        self.pendingConfidence = nil
        self.pendingIou = nil
        completion?(.success(self))
      case .failure(let error):
        print("Failed to load model with error: \(error)")
        completion?(.failure(error))
      }
    }

    switch task {
    case .classify:
      Classifier.create(unwrappedModelURL: modelURL, completion: handleResult)
    case .segment:
      Segmenter.create(unwrappedModelURL: modelURL, completion: handleResult)
    case .pose:
      PoseEstimator.create(unwrappedModelURL: modelURL, completion: handleResult)
    case .obb:
      ObbDetector.create(unwrappedModelURL: modelURL, completion: handleResult)
    default:
      ObjectDetector.create(unwrappedModelURL: modelURL, completion: handleResult)

    }
  }

  // MARK: - Threshold Configuration Methods

  /// Sets the maximum number of detection items to include in results.
  /// - Parameter numItems: The maximum number of items to include (default is 30).
  public func setNumItemsThreshold(_ numItems: Int) {
    pendingNumItems = numItems
    (predictor as? BasePredictor)?.setNumItemsThreshold(numItems: numItems)
  }

  /// Gets the current maximum number of detection items.
  /// - Returns: The current threshold value, or nil if not applicable.
  public func getNumItemsThreshold() -> Int? {
    (predictor as? BasePredictor)?.numItemsThreshold ?? pendingNumItems
  }

  /// Sets the confidence threshold for filtering results.
  /// - Parameter confidence: The confidence threshold value (0.0 to 1.0, default is 0.25).
  public func setConfidenceThreshold(_ confidence: Double) {
    guard (0.0...1.0).contains(confidence) else {
      print("Warning: Confidence threshold should be between 0.0 and 1.0")
      return
    }
    pendingConfidence = confidence
    (predictor as? BasePredictor)?.setConfidenceThreshold(confidence: confidence)
  }

  /// Gets the current confidence threshold.
  /// - Returns: The current threshold value, or nil if not applicable.
  public func getConfidenceThreshold() -> Double? {
    (predictor as? BasePredictor)?.confidenceThreshold ?? pendingConfidence
  }

  /// Sets the IoU (Intersection over Union) threshold for non-maximum suppression.
  /// - Parameter iou: The IoU threshold value (0.0 to 1.0, default is 0.4).
  public func setIouThreshold(_ iou: Double) {
    guard (0.0...1.0).contains(iou) else {
      print("Warning: IoU threshold should be between 0.0 and 1.0")
      return
    }
    pendingIou = iou
    (predictor as? BasePredictor)?.setIouThreshold(iou: iou)
  }

  /// Gets the current IoU threshold.
  /// - Returns: The current threshold value, or nil if not applicable.
  public func getIouThreshold() -> Double? {
    (predictor as? BasePredictor)?.iouThreshold ?? pendingIou
  }

  /// Sets all thresholds at once.
  /// - Parameters:
  ///   - numItems: The maximum number of items to include.
  ///   - confidence: The confidence threshold value (0.0 to 1.0).
  ///   - iou: The IoU threshold value (0.0 to 1.0).
  public func setThresholds(numItems: Int? = nil, confidence: Double? = nil, iou: Double? = nil) {
    numItems.map { setNumItemsThreshold($0) }
    confidence.map { setConfidenceThreshold($0) }
    iou.map { setIouThreshold($0) }
  }

  public func callAsFunction(_ uiImage: UIImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    let ciImage = CIImage(image: uiImage)!
    let result = predictor.predictOnImage(image: ciImage)
    //        if returnAnnotatedImage {
    //            let annotatedImage = drawYOLODetections(on: ciImage, result: result)
    //            result.annotatedImage = annotatedImage
    //        }
    return result
  }

  public func callAsFunction(_ ciImage: CIImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    let result = predictor.predictOnImage(image: ciImage)
    //    if returnAnnotatedImage {
    //      let annotatedImage = drawYOLODetections(on: ciImage, result: result)
    //      result.annotatedImage = annotatedImage
    //    }
    return result
  }

  public func callAsFunction(_ cgImage: CGImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    let ciImage = CIImage(cgImage: cgImage)
    let result = predictor.predictOnImage(image: ciImage)
    //    if returnAnnotatedImage {
    //      let annotatedImage = drawYOLODetections(on: ciImage, result: result)
    //      result.annotatedImage = annotatedImage
    //    }
    return result
  }

  public func callAsFunction(
    _ resourceName: String,
    withExtension ext: String? = nil,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    guard let url = Bundle.main.url(forResource: resourceName, withExtension: ext),
      let data = try? Data(contentsOf: url),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  public func callAsFunction(
    _ remoteURL: URL?,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    guard let remoteURL = remoteURL,
      let data = try? Data(contentsOf: remoteURL),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  public func callAsFunction(
    _ localPath: String,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    let fileURL = URL(fileURLWithPath: localPath)
    guard let data = try? Data(contentsOf: fileURL),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  @MainActor @available(iOS 16.0, *)
  public func callAsFunction(
    _ swiftUIImage: SwiftUI.Image,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    let renderer = ImageRenderer(content: swiftUIImage)
    guard let uiImage = renderer.uiImage else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }
}
