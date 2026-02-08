// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Vision
import XCTest

@testable import YOLO

/// Minimal tests for BasePredictor functionality
class BasePredictorTests: XCTestCase {

  func testInitialization() {
    // Test BasePredictor initialization
    let predictor = BasePredictor()

    XCTAssertFalse(predictor.isModelLoaded)
    XCTAssertEqual(predictor.labels.count, 0)
    XCTAssertEqual(predictor.confidenceThreshold, 0.25, accuracy: 0.001)
    XCTAssertEqual(predictor.iouThreshold, 0.7, accuracy: 0.001)
    XCTAssertEqual(predictor.numItemsThreshold, 30)
    XCTAssertFalse(predictor.isUpdating)
    XCTAssertNil(predictor.currentBuffer)
  }

  func testConfidenceThresholdSetting() {
    // Test confidence threshold configuration
    let predictor = BasePredictor()

    predictor.setConfidenceThreshold(confidence: 0.8)
    XCTAssertEqual(predictor.confidenceThreshold, 0.8, accuracy: 0.001)

    predictor.setConfidenceThreshold(confidence: 0.1)
    XCTAssertEqual(predictor.confidenceThreshold, 0.1, accuracy: 0.001)
  }

  func testIoUThresholdSetting() {
    // Test IoU threshold configuration
    let predictor = BasePredictor()

    predictor.setIouThreshold(iou: 0.7)
    XCTAssertEqual(predictor.iouThreshold, 0.7, accuracy: 0.001)

    predictor.setIouThreshold(iou: 0.2)
    XCTAssertEqual(predictor.iouThreshold, 0.2, accuracy: 0.001)
  }

  func testNumItemsThresholdSetting() {
    // Test number of items threshold configuration
    let predictor = BasePredictor()

    predictor.setNumItemsThreshold(numItems: 50)
    XCTAssertEqual(predictor.numItemsThreshold, 50)

    predictor.setNumItemsThreshold(numItems: 10)
    XCTAssertEqual(predictor.numItemsThreshold, 10)
  }

  func testBasePredictOnImage() {
    // Test base predictOnImage method returns empty result
    let predictor = BasePredictor()
    let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))

    let result = predictor.predictOnImage(image: image)

    XCTAssertEqual(result.orig_shape, .zero)
    XCTAssertEqual(result.boxes.count, 0)
    XCTAssertEqual(result.speed, 0, accuracy: 0.001)
    XCTAssertEqual(result.names.count, 0)
  }

  func testBaseProcessObservations() {
    // Test base processObservations method does nothing
    let predictor = BasePredictor()

    // Should not crash when called
    predictor.processObservations(for: MockVNRequest(), error: nil)
    predictor.processObservations(for: MockVNRequest(), error: NSError(domain: "test", code: 0))
  }

  func testLabelsProperty() {
    // Test labels property can be read and written
    let predictor = BasePredictor()
    let testLabels = ["person", "car", "dog"]

    predictor.labels = testLabels
    XCTAssertEqual(predictor.labels, testLabels)

    predictor.labels.append("cat")
    XCTAssertEqual(predictor.labels.count, 4)
    XCTAssertEqual(predictor.labels.last, "cat")
  }

  func testIsUpdatingFlag() {
    // Test isUpdating flag can be set and read
    let predictor = BasePredictor()

    XCTAssertFalse(predictor.isUpdating)

    predictor.isUpdating = true
    XCTAssertTrue(predictor.isUpdating)

    predictor.isUpdating = false
    XCTAssertFalse(predictor.isUpdating)
  }

  func testModelInputSizeInitialization() {
    // Test model input size has proper default values
    let predictor = BasePredictor()

    XCTAssertEqual(predictor.modelInputSize.width, 0)
    XCTAssertEqual(predictor.modelInputSize.height, 0)
  }

  func testTimingProperties() {
    // Test timing properties initialization
    let predictor = BasePredictor()

    XCTAssertEqual(predictor.t0, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t1, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t2, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t4, 1.0, accuracy: 0.001)  // non-zero to avoid infinity FPS on first frame
    XCTAssertGreaterThan(predictor.t3, 0)  // Should be initialized with current time
  }
}

// MARK: - Mock Classes

class MockVNRequest: VNRequest, @unchecked Sendable {
  init() {
    super.init(completionHandler: nil)
  }

  @available(*, unavailable)
  required init?(coder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }
}
