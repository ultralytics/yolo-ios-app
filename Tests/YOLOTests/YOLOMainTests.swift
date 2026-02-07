// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import XCTest

@testable import YOLOCore

/// Tests for YOLO main class functionality
class YOLOMainTests: XCTestCase {

  func testYOLOInitializationWithInvalidPath() async {
    do {
      _ = try await YOLO("invalid_model_path", task: .detect)
      XCTFail("Should not succeed with invalid path")
    } catch {
      XCTAssertTrue(error is PredictorError)
      if case PredictorError.modelFileNotFound = error {
        // Expected
      } else {
        XCTFail("Unexpected error type: \(error)")
      }
    }
  }

  func testYOLOAllTaskTypes() async {
    let tasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]

    for task in tasks {
      do {
        _ = try await YOLO("invalid_path", task: task)
        XCTFail("Should not succeed with invalid path for task \(task)")
      } catch {
        XCTAssertNotNil(error)
      }
    }
  }

  func testYOLOConfigurationDefaults() {
    let config = YOLOConfiguration()
    XCTAssertEqual(config.confidenceThreshold, 0.25, accuracy: 0.001)
    XCTAssertEqual(config.iouThreshold, 0.45, accuracy: 0.001)
    XCTAssertEqual(config.maxDetections, 30)
  }

  func testYOLOConfigurationCustom() {
    let config = YOLOConfiguration(
      confidenceThreshold: 0.5,
      iouThreshold: 0.7,
      maxDetections: 50
    )
    XCTAssertEqual(config.confidenceThreshold, 0.5, accuracy: 0.001)
    XCTAssertEqual(config.iouThreshold, 0.7, accuracy: 0.001)
    XCTAssertEqual(config.maxDetections, 50)
  }
}
