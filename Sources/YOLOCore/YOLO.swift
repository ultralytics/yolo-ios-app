// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import Foundation

#if canImport(UIKit)
  import UIKit
#endif

/// The primary interface for working with YOLO models.
///
/// ```swift
/// let model = try await YOLO("yolo26n", task: .detect)
/// let result = model(cgImage)
/// ```
public final class YOLO: @unchecked Sendable {
  /// The underlying predictor for inference.
  public private(set) var predictor: Predictor

  /// The task this model performs.
  public let task: YOLOTask

  /// Current configuration for thresholds.
  public var configuration: YOLOConfiguration {
    get { predictor.configuration }
    set { predictor.configuration = newValue }
  }

  private init(predictor: Predictor, task: YOLOTask) {
    self.predictor = predictor
    self.task = task
  }

  /// Initialize YOLO with a local model name or path.
  public init(_ modelPathOrName: String, task: YOLOTask) async throws {
    self.task = task

    let modelURL: URL? = YOLO.resolveModelURL(modelPathOrName)
    guard let url = modelURL else {
      throw PredictorError.modelFileNotFound
    }

    self.predictor = try await YOLO.createPredictor(for: task, modelURL: url)
  }

  /// Initialize YOLO with a remote URL for automatic download and caching.
  public init(url: URL, task: YOLOTask) async throws {
    self.task = task

    let downloader = ModelDownloader()
    let modelPath = try await downloader.download(from: url, task: task)
    self.predictor = try await YOLO.createPredictor(for: task, modelURL: modelPath)
  }

  // MARK: - callAsFunction overloads

  /// Run inference on a CGImage.
  public func callAsFunction(_ cgImage: CGImage) -> YOLOResult {
    let ciImage = CIImage(cgImage: cgImage)
    return predictor.predictOnImage(image: ciImage)
  }

  /// Run inference on a CIImage.
  public func callAsFunction(_ ciImage: CIImage) -> YOLOResult {
    predictor.predictOnImage(image: ciImage)
  }

  #if canImport(UIKit)
    /// Run inference on a UIImage.
    public func callAsFunction(_ uiImage: UIImage) -> YOLOResult {
      guard let ciImage = CIImage(image: uiImage) else {
        return YOLOResult(orig_shape: .zero, boxes: [], names: [])
      }
      return predictor.predictOnImage(image: ciImage)
    }
  #endif

  /// Run inference on a CVPixelBuffer (for real-time camera frames).
  public func callAsFunction(_ pixelBuffer: CVPixelBuffer) -> YOLOResult {
    predictor.predict(pixelBuffer: pixelBuffer)
  }

  // MARK: - Private

  private static func resolveModelURL(_ modelPathOrName: String) -> URL? {
    let lowercasedPath = modelPathOrName.lowercased()

    if lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage")
      || lowercasedPath.hasSuffix(".mlmodelc")
    {
      let possibleURL = URL(fileURLWithPath: modelPathOrName)
      if FileManager.default.fileExists(atPath: possibleURL.path) {
        return possibleURL
      }
    }

    if let compiledURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc") {
      return compiledURL
    }
    if let packageURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlpackage") {
      return packageURL
    }
    return nil
  }

  private static func createPredictor(for task: YOLOTask, modelURL: URL) async throws -> Predictor {
    switch task {
    case .classify:
      return try await Classifier.create(modelURL: modelURL)
    case .segment:
      return try await Segmenter.create(modelURL: modelURL)
    case .pose:
      return try await PoseEstimator.create(modelURL: modelURL)
    case .obb:
      return try await ObbDetector.create(modelURL: modelURL)
    case .detect:
      return try await ObjectDetector.create(modelURL: modelURL)
    }
  }
}
