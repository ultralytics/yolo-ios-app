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
///
/// Model loading is asynchronous. Calling a `callAsFunction` overload before the init completion
/// handler has fired returns an empty `YOLOResult`. Use `isLoaded` to check readiness, or perform
/// inference from inside the completion handler.
public final class YOLO: @unchecked Sendable {
  var predictor: Predictor?
  private var modelDownloader: YOLOModelDownloader?

  private var pendingNumItems: Int?
  private var pendingConfidence: Double?
  private var pendingIou: Double?

  /// Whether the model has finished loading and is ready to run inference.
  ///
  /// `false` while the model is still compiling/loading (or if loading failed).
  /// `true` once the completion handler passed to `init` has fired with `.success`.
  public var isLoaded: Bool {
    (predictor as? BasePredictor)?.isModelLoaded ?? false
  }

  /// Initialize YOLO with remote URL for automatic download and caching
  public init(url: URL, task: YOLOTask, completion: @escaping (Result<YOLO, Error>) -> Void) {
    modelDownloader = YOLOModelDownloader()
    modelDownloader?.download(from: url, task: task) { [weak self] result in
      guard let self = self else { return }
      self.modelDownloader = nil
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
    guard let modelURL = ModelPathResolver.resolve(modelPathOrName) else {
      completion?(.failure(PredictorError.modelFileNotFound))
      return
    }
    loadModel(from: modelURL, task: task, completion: completion)
  }

  /// Load model from URL with task-specific predictor creation
  private func loadModel(
    from modelURL: URL, task: YOLOTask, completion: ((Result<YOLO, Error>) -> Void)?
  ) {
    BasePredictor.create(for: task, modelURL: modelURL) { [weak self] result in
      guard let self = self else { return }
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
        YOLOLog.error("Failed to load model: \(error)")
        completion?(.failure(error))
      }
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
    guard validateUnitRange(confidence, name: "Confidence threshold") else { return }
    pendingConfidence = confidence
    (predictor as? BasePredictor)?.setConfidenceThreshold(confidence: confidence)
  }

  /// Gets the current confidence threshold.
  /// - Returns: The current threshold value, or nil if not applicable.
  public func getConfidenceThreshold() -> Double? {
    (predictor as? BasePredictor)?.confidenceThreshold ?? pendingConfidence
  }

  /// Sets the IoU (Intersection over Union) threshold for non-maximum suppression.
  /// - Parameter iou: The IoU threshold value (0.0 to 1.0, default is 0.7).
  public func setIouThreshold(_ iou: Double) {
    guard validateUnitRange(iou, name: "IoU threshold") else { return }
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

  /// Runs inference against the loaded predictor, or returns an empty result if no model is loaded.
  private func run(_ ciImage: CIImage) -> YOLOResult {
    predictor?.predictOnImage(image: ciImage) ?? .empty
  }

  public func callAsFunction(_ uiImage: UIImage) -> YOLOResult {
    // CIImage(image:) drops UIImage.imageOrientation, so non-`.up` photos (e.g. portrait
    // shots with orientation = .right) would otherwise be inferred against raw, rotated
    // pixels. Build the CIImage from the backing CGImage and re-apply the orientation.
    if let cgImage = uiImage.cgImage {
      let orientation = CGImagePropertyOrientation(uiImage.imageOrientation)
      return run(CIImage(cgImage: cgImage).oriented(orientation))
    }
    let upright = uiImage.uprightForYOLO()
    if let cgImage = upright.cgImage { return run(CIImage(cgImage: cgImage)) }
    if let ciImage = upright.ciImage ?? CIImage(image: upright) { return run(ciImage) }
    return .empty
  }

  public func callAsFunction(_ ciImage: CIImage) -> YOLOResult {
    return run(ciImage)
  }

  public func callAsFunction(_ cgImage: CGImage) -> YOLOResult {
    return run(CIImage(cgImage: cgImage))
  }

  public func callAsFunction(
    _ resourceName: String,
    withExtension ext: String? = nil
  ) -> YOLOResult {
    guard let url = Bundle.main.url(forResource: resourceName, withExtension: ext),
      let data = try? Data(contentsOf: url),
      let uiImage = UIImage(data: data)
    else { return .empty }
    return self(uiImage)
  }

  /// Runs inference on an image fetched from a remote URL.
  ///
  /// - Warning: This overload performs a **synchronous** network fetch on the caller's thread
  ///   via `Data(contentsOf:)` and may block indefinitely. Prefer downloading the bytes yourself
  ///   (e.g. `URLSession`) and calling `callAsFunction(_: UIImage)` / `callAsFunction(_: CIImage)`.
  @available(
    *, deprecated,
    message:
      "Blocking network I/O on the calling thread. Fetch the image yourself and call callAsFunction(_: UIImage) instead."
  )
  public func callAsFunction(
    _ remoteURL: URL?
  ) -> YOLOResult {
    guard let remoteURL = remoteURL,
      let data = try? Data(contentsOf: remoteURL),
      let uiImage = UIImage(data: data)
    else { return .empty }
    return self(uiImage)
  }

  public func callAsFunction(
    _ localPath: String
  ) -> YOLOResult {
    let fileURL = URL(fileURLWithPath: localPath)
    guard let data = try? Data(contentsOf: fileURL),
      let uiImage = UIImage(data: data)
    else { return .empty }
    return self(uiImage)
  }

  @MainActor @available(iOS 16.0, *)
  public func callAsFunction(
    _ swiftUIImage: SwiftUI.Image
  ) -> YOLOResult {
    let renderer = ImageRenderer(content: swiftUIImage)
    guard let uiImage = renderer.uiImage else { return .empty }
    return self(uiImage)
  }
}

/// Validates that `value` lies in the unit interval and logs a warning otherwise.
@inline(__always)
func validateUnitRange(_ value: Double, name: String) -> Bool {
  guard (0.0...1.0).contains(value) else {
    YOLOLog.warning("\(name) should be between 0.0 and 1.0 (got \(value))")
    return false
  }
  return true
}
