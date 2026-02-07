// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Vision
import XCTest

@testable import YOLOCore

/// Tests for BasePredictor functionality
class BasePredictorTests: XCTestCase {

  func testInitialization() {
    let predictor = BasePredictor()

    XCTAssertFalse(predictor.isModelLoaded)
    XCTAssertEqual(predictor.labels.count, 0)
    XCTAssertTrue(predictor.requiresNMS)
    XCTAssertEqual(predictor.configuration.confidenceThreshold, 0.25, accuracy: 0.001)
    XCTAssertEqual(predictor.configuration.iouThreshold, 0.45, accuracy: 0.001)
    XCTAssertEqual(predictor.configuration.maxDetections, 30)
  }

  func testConfigurationUpdate() {
    let predictor = BasePredictor()

    predictor.configuration.confidenceThreshold = 0.8
    XCTAssertEqual(predictor.configuration.confidenceThreshold, 0.8, accuracy: 0.001)

    predictor.configuration.iouThreshold = 0.3
    XCTAssertEqual(predictor.configuration.iouThreshold, 0.3, accuracy: 0.001)

    predictor.configuration.maxDetections = 50
    XCTAssertEqual(predictor.configuration.maxDetections, 50)
  }

  func testBasePredictOnImage() {
    let predictor = BasePredictor()
    let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))

    let result = predictor.predictOnImage(image: image)

    XCTAssertEqual(result.boxes.count, 0)
    XCTAssertEqual(result.names.count, 0)
  }

  func testLabelsProperty() {
    let predictor = BasePredictor()
    XCTAssertEqual(predictor.labels.count, 0)
  }

  func testModelInputSizeInitialization() {
    let predictor = BasePredictor()
    XCTAssertEqual(predictor.modelInputSize.width, 0)
    XCTAssertEqual(predictor.modelInputSize.height, 0)
  }

  func testTimingProperties() {
    let predictor = BasePredictor()

    XCTAssertEqual(predictor.t0, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t1, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t2, 0.0, accuracy: 0.001)
    XCTAssertEqual(predictor.t4, 0.0, accuracy: 0.001)
    XCTAssertGreaterThan(predictor.t3, 0)
  }
}
