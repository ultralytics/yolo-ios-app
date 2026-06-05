// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import CoreImage
import UIKit
import XCTest

@testable import UltralyticsYOLO

/// Minimal tests for YOLO main class functionality
class YOLOMainTests: XCTestCase {

  func testYOLOInitializationWithInvalidPath() {
    let expectation = XCTestExpectation(description: "Invalid model path")

    let _ = YOLO("invalid_model_path", task: .detect) { result in
      switch result {
      case .success(_):
        XCTFail("Should not succeed with invalid path")
      case .failure(let error):
        XCTAssertNotNil(error)
        if case PredictorError.modelFileNotFound = error {
          // Expected error type
        } else {
          XCTFail("Unexpected error type: \(error)")
        }
        expectation.fulfill()
      }
    }

    wait(for: [expectation], timeout: 5.0)
  }

  func testYOLOCallAsFunctionWithUIImage() {
    let yolo = YOLO("invalid_model_path", task: .detect)
    let predictor = MockPredictor()
    yolo.predictor = predictor

    let image = makeTestImage(size: CGSize(width: 12, height: 8), color: .red)

    let result = yolo(image)

    XCTAssertEqual(result.orig_shape, CGSize(width: 12, height: 8))
    XCTAssertEqual(predictor.callCount, 1)
  }

  func testYOLOCallAsFunctionRoutesCIAndCGImages() {
    let yolo = YOLO("invalid_model_path", task: .detect)
    let predictor = MockPredictor()
    yolo.predictor = predictor

    let ciImage = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 10))
    XCTAssertEqual(yolo(ciImage).orig_shape, CGSize(width: 20, height: 10))

    let cgImage = makeTestImage(size: CGSize(width: 7, height: 5), color: .green).cgImage!
    XCTAssertEqual(yolo(cgImage).orig_shape, CGSize(width: 7, height: 5))
    XCTAssertEqual(predictor.callCount, 2)
  }

  func testYOLOCallAsFunctionHandlesMissingImageSources() {
    let yolo = YOLO("invalid_model_path", task: .detect)

    XCTAssertEqual(yolo("missing-resource", withExtension: "png").orig_shape, .zero)
    XCTAssertEqual(yolo("/tmp/definitely-missing-yolo-image.png").orig_shape, .zero)
  }

  func testYOLOCallAsFunctionLoadsLocalImagePath() throws {
    let yolo = YOLO("invalid_model_path", task: .detect)
    let predictor = MockPredictor()
    yolo.predictor = predictor

    let image = makeTestImage(size: CGSize(width: 11, height: 9), color: .white)
    let imagePath = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
      .appendingPathExtension("png")
    try image.pngData()?.write(to: imagePath)
    defer { try? FileManager.default.removeItem(at: imagePath) }

    XCTAssertEqual(yolo(imagePath.path).orig_shape, CGSize(width: 11, height: 9))
  }

  func testYOLOAllTaskTypes() {
    let tasks: [YOLOTask] = [.detect, .segment, .semantic, .classify, .pose, .obb]

    for task in tasks {
      let expectation = XCTestExpectation(description: "Task \(task)")

      let _ = YOLO("invalid_path", task: task) { result in
        switch result {
        case .success(_):
          XCTFail("Should not succeed with invalid path for task \(task)")
        case .failure(let error):
          XCTAssertNotNil(error)
          expectation.fulfill()
        }
      }

      wait(for: [expectation], timeout: 3.0)
    }
  }

  func testYOLOThresholdsValidateAndApplyToLoadedPredictor() {
    let yolo = YOLO("invalid_model_path", task: .detect)
    let predictor = BasePredictor()
    yolo.predictor = predictor

    yolo.setThresholds(numItems: 3, confidence: 0.4, iou: 0.6)

    XCTAssertEqual(yolo.getNumItemsThreshold(), 3)
    XCTAssertEqual(yolo.getConfidenceThreshold() ?? -1, 0.4, accuracy: 0.001)
    XCTAssertEqual(yolo.getIouThreshold() ?? -1, 0.6, accuracy: 0.001)
    XCTAssertTrue(validateUnitRange(0, name: "threshold"))
    XCTAssertTrue(validateUnitRange(1, name: "threshold"))
    XCTAssertFalse(validateUnitRange(-0.1, name: "threshold"))
    XCTAssertFalse(validateUnitRange(1.1, name: "threshold"))
  }

  func testModelCacheValidatesPackageManifestsAndSizesEntries() throws {
    let cache = YOLOModelCache.shared
    let sourceURL = URL(string: "https://example.com/\(UUID().uuidString).zip")!
    let key = cache.cacheKey(for: sourceURL, task: .detect)
    let packageURL = cache.cacheDirectory.appendingPathComponent(key).appendingPathExtension(
      "mlpackage")
    defer { try? FileManager.default.removeItem(at: packageURL) }

    try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: true)
    XCTAssertNil(cache.getCachedModelPath(url: sourceURL, task: .detect))

    try Data("{}".utf8).write(to: packageURL.appendingPathComponent("Manifest.json"))
    try Data(repeating: 1, count: 16).write(to: packageURL.appendingPathComponent("weights.bin"))

    XCTAssertEqual(cache.getCachedModelPath(url: sourceURL, task: .detect), packageURL)
    XCTAssertTrue(try cache.listCachedModels().contains(key))
    XCTAssertGreaterThanOrEqual(try cache.getCacheSize(), 18)
    XCTAssertNotEqual(cache.cacheKey(for: sourceURL, task: .detect), cache.cacheKey(for: sourceURL))
  }

  func testLoggerEvaluatesMessages() {
    var emitted = 0
    func message(_ text: String) -> String {
      emitted += 1
      return text
    }

    YOLOLog.info(message("info"))
    YOLOLog.warning(message("warning"))
    YOLOLog.error(message("error"))

    XCTAssertEqual(emitted, 3)
  }

}

private func makeTestImage(size: CGSize, color: UIColor) -> UIImage {
  let format = UIGraphicsImageRendererFormat()
  format.scale = 1
  return UIGraphicsImageRenderer(size: size, format: format).image { context in
    color.setFill()
    context.fill(CGRect(origin: .zero, size: size))
  }
}

// MARK: - Mock Classes for Testing

class MockPredictor: Predictor, @unchecked Sendable {
  var labels: [String] = []
  var isUpdating: Bool = false
  var callCount = 0

  func predict(
    sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?,
    onInferenceTime: InferenceTimeListener?
  ) {
    // Mock implementation - do nothing
  }

  func predictOnImage(image: CIImage) -> YOLOResult {
    callCount += 1
    return YOLOResult(orig_shape: image.extent.size, boxes: [], speed: 0, names: labels)
  }
}

// Extension to allow creating YOLO instances without going through the full init
extension YOLO {
  static func __allocating_init() -> YOLO {
    // Create a simple mock YOLO instance
    // Note: This will actually try to initialize with an invalid path
    // but we handle the error in the test
    return YOLO("mock_model", task: .detect) { _ in }
  }
}
