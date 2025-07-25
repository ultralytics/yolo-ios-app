// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import CoreML
import Vision
import CoreImage
@testable import YOLO

/// Unit tests for Classifier that don't require actual models
/// Focus on testable logic and data processing
class ClassifierUnitTests: XCTestCase {
    
    var classifier: Classifier!
    
    override func setUp() {
        super.setUp()
        classifier = Classifier()
    }
    
    override func tearDown() {
        classifier = nil
        super.tearDown()
    }
    
    // MARK: - Property Tests
    
    func testInitialState() {
        XCTAssertEqual(classifier.labels.count, 0)
        XCTAssertNil(classifier.visionRequest)
        XCTAssertFalse(classifier.isUpdating)
        XCTAssertEqual(classifier.inputSize, CGSize(width: 640, height: 640))
    }
    
    func testLabelManagement() {
        let testLabels = ["cat", "dog", "bird", "fish"]
        classifier.labels = testLabels
        
        XCTAssertEqual(classifier.labels, testLabels)
        XCTAssertEqual(classifier.labels.count, 4)
    }
    
    // MARK: - Data Processing Tests
    
    func testEmptyPrediction() {
        // Test behavior with no model loaded
        let image = createTestImage()
        let result = classifier.predictOnImage(image: image)
        
        // Should return default result
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
        XCTAssertNil(result.probs)
        XCTAssertEqual(result.orig_shape.width, 640)
        XCTAssertEqual(result.orig_shape.height, 640)
    }
    
    func testResultProcessing() {
        // Test the logic of processing classification results
        classifier.labels = ["apple", "banana", "orange", "grape", "strawberry"]
        
        // Simulate processing logic without actual Vision framework
        let confidences = [0.05, 0.1, 0.6, 0.2, 0.05]
        let topIndex = confidences.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        let topLabel = classifier.labels[topIndex]
        let topConfidence = confidences[topIndex]
        
        XCTAssertEqual(topLabel, "orange")
        XCTAssertEqual(topConfidence, 0.6, accuracy: 0.001)
    }
    
    // MARK: - Thread Safety
    
    func testConcurrentLabelAccess() {
        let expectation = XCTestExpectation(description: "Concurrent label access")
        expectation.expectedFulfillmentCount = 10
        
        let labels = ["label1", "label2", "label3"]
        
        for _ in 0..<10 {
            DispatchQueue.global().async {
                self.classifier.labels = labels
                _ = self.classifier.labels
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
    
    // MARK: - Helper Methods
    
    private func createTestImage(width: CGFloat = 640, height: CGFloat = 640) -> CIImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: Int(width),
            height: Int(height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return CIImage()
        }
        
        // Create a simple test pattern
        context.setFillColor(UIColor.gray.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let cgImage = context.makeImage() else {
            return CIImage()
        }
        
        return CIImage(cgImage: cgImage)
    }
}