// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import CoreML
@testable import YOLO

/// Minimal tests for ThresholdProvider
class ThresholdProviderTests: XCTestCase {
    
    func testDefaultInitialization() {
        // Test ThresholdProvider with default values
        let provider = ThresholdProvider()
        
        XCTAssertEqual(provider.featureNames.count, 2)
        XCTAssertTrue(provider.featureNames.contains("iouThreshold"))
        XCTAssertTrue(provider.featureNames.contains("confidenceThreshold"))
        
        let iouValue = provider.featureValue(for: "iouThreshold")
        let confValue = provider.featureValue(for: "confidenceThreshold")
        
        XCTAssertNotNil(iouValue)
        XCTAssertNotNil(confValue)
        XCTAssertEqual(iouValue?.doubleValue, 0.45, accuracy: 0.001)
        XCTAssertEqual(confValue?.doubleValue, 0.25, accuracy: 0.001)
    }
    
    func testCustomInitialization() {
        // Test ThresholdProvider with custom values
        let provider = ThresholdProvider(iouThreshold: 0.7, confidenceThreshold: 0.8)
        
        let iouValue = provider.featureValue(for: "iouThreshold")
        let confValue = provider.featureValue(for: "confidenceThreshold")
        
        XCTAssertEqual(iouValue?.doubleValue, 0.7, accuracy: 0.001)
        XCTAssertEqual(confValue?.doubleValue, 0.8, accuracy: 0.001)
    }
    
    func testInvalidFeatureName() {
        // Test behavior with invalid feature name
        let provider = ThresholdProvider()
        let value = provider.featureValue(for: "invalidFeature")
        
        XCTAssertNil(value)
    }
    
    func testFeatureNames() {
        // Test featureNames property
        let provider = ThresholdProvider()
        let names = provider.featureNames
        
        XCTAssertEqual(names.count, 2)
        XCTAssertTrue(names.isSubset(of: ["iouThreshold", "confidenceThreshold"]))
    }
}
