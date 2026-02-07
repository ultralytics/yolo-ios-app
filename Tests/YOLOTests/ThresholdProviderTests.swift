// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import XCTest

@testable import YOLOCore

/// Tests for ThresholdProvider
class ThresholdProviderTests: XCTestCase {

  func testDefaultInitialization() {
    let provider = ThresholdProvider()

    XCTAssertEqual(provider.featureNames.count, 2)
    XCTAssertTrue(provider.featureNames.contains("iouThreshold"))
    XCTAssertTrue(provider.featureNames.contains("confidenceThreshold"))

    let iouValue = provider.featureValue(for: "iouThreshold")
    let confValue = provider.featureValue(for: "confidenceThreshold")

    XCTAssertNotNil(iouValue)
    XCTAssertNotNil(confValue)
    XCTAssertEqual(iouValue!.doubleValue, 0.45, accuracy: 0.001)
    XCTAssertEqual(confValue!.doubleValue, 0.25, accuracy: 0.001)
  }

  func testCustomInitialization() {
    let provider = ThresholdProvider(iouThreshold: 0.7, confidenceThreshold: 0.8)

    let iouValue = provider.featureValue(for: "iouThreshold")
    let confValue = provider.featureValue(for: "confidenceThreshold")

    XCTAssertEqual(iouValue!.doubleValue, 0.7, accuracy: 0.001)
    XCTAssertEqual(confValue!.doubleValue, 0.8, accuracy: 0.001)
  }

  func testInvalidFeatureName() {
    let provider = ThresholdProvider()
    let value = provider.featureValue(for: "invalidFeature")
    XCTAssertNil(value)
  }

  func testFeatureNames() {
    let provider = ThresholdProvider()
    let names = provider.featureNames

    XCTAssertEqual(names.count, 2)
    XCTAssertTrue(names.isSubset(of: ["iouThreshold", "confidenceThreshold"]))
  }
}
