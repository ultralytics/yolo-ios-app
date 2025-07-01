// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import CoreML
import Vision
import CoreImage
@testable import YOLO

/// Tests for the Classifier model implementation
class ClassifierTests: XCTestCase {
    
    var classifier: Classifier!
    
    override func setUp() {
        super.setUp()
        classifier = Classifier()
    }
    
    override func tearDown() {
        classifier = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testClassifierInitialization() {
        // Test that classifier is properly initialized
        XCTAssertNotNil(classifier)
        XCTAssertEqual(classifier.labels.count, 0)
        XCTAssertNil(classifier.visionRequest)
        XCTAssertFalse(classifier.isUpdating)
    }
    
    // MARK: - Label Management Tests
    
    func testLabelsProperty() {
        // Test setting and getting labels
        let testLabels = ["cat", "dog", "bird"]
        classifier.labels = testLabels
        XCTAssertEqual(classifier.labels, testLabels)
    }
    
    // MARK: - Predict on Image Tests
    
    func testPredictOnImageWithNoModel() {
        // Test prediction when no model is loaded
        classifier.labels = ["cat", "dog", "bird"]
        let image = createTestImage()
        
        let result = classifier.predictOnImage(image: image)
        
        // Without a loaded model, result should have default values
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
        XCTAssertEqual(result.orig_shape?.width, 640, accuracy: 0.001)
        XCTAssertEqual(result.orig_shape?.height, 640, accuracy: 0.001)
    }
    
    func testPredictOnImageWithMLMultiArray() {
        // Test prediction with MLMultiArray results
        classifier.labels = ["apple", "banana", "orange", "grape", "strawberry"]
        
        let mockMultiArray = createMockMLMultiArray(values: [0.05, 0.1, 0.6, 0.2, 0.05])
        let observation = MockVNCoreMLFeatureValueObservation(multiArray: mockMultiArray)
        let mockRequest = MockVNCoreMLRequestForTesting(results: [observation])
        classifier.visionRequest = mockRequest
        
        let image = createTestImage()
        let result = classifier.predictOnImage(image: image)
        
        XCTAssertNotNil(result.probs)
        XCTAssertEqual(result.probs?.top1, "orange")
        XCTAssertEqual(result.probs?.top1Conf ?? 0, 0.6, accuracy: 0.001)
        XCTAssertNotNil(result.annotatedImage)
        XCTAssertEqual(result.names, classifier.labels)
        // Should use default size when inputSize is nil
        XCTAssertEqual(result.orig_shape.width, 640)
        XCTAssertEqual(result.orig_shape.height, 640)
    }
    
    func testPredictOnImageSetsInputSize() {
        // Test that predictOnImage behavior without a model
        let image = createTestImage()
        _ = classifier.predictOnImage(image: image)
        
        // Without a model loaded, inputSize remains nil
        XCTAssertNil(classifier.inputSize)
    }
    
    func testPredictOnImageWithLargeImage() {
        // Test prediction with large image
        let image = createTestImage(width: 4000, height: 3000)
        
        let result = classifier.predictOnImage(image: image)
        
        XCTAssertEqual(result.orig_shape?.width, 640, accuracy: 0.001)
        XCTAssertEqual(result.orig_shape?.height, 640, accuracy: 0.001)
    }
    
    // MARK: - Listener Tests
    
    func testResultsListener() {
        // Test that results listener can be set
        let mockListener = MockResultsListener()
        classifier.currentOnResultsListener = mockListener
        XCTAssertNotNil(classifier.currentOnResultsListener)
    }
    
    func testInferenceTimeListener() {
        // Test that inference time listener can be set
        let mockListener = MockInferenceTimeListener()
        classifier.currentOnInferenceTimeListener = mockListener
        XCTAssertNotNil(classifier.currentOnInferenceTimeListener)
    }
    
    // MARK: - Thread Safety Tests
    
    func testConcurrentPredictions() {
        // Test that classifier can handle concurrent predictions safely
        let expectation = XCTestExpectation(description: "Concurrent predictions")
        expectation.expectedFulfillmentCount = 10
        
        let group = DispatchGroup()
        
        for i in 0..<10 {
            group.enter()
            DispatchQueue.global().async {
                let image = self.createTestImage()
                let result = self.classifier.predictOnImage(image: image)
                
                XCTAssertNotNil(result)
                XCTAssertEqual(result.boxes.count, 0)
                
                group.leave()
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    // MARK: - Edge Cases
    
    func testPredictOnImageWithEmptyLabels() {
        // Test prediction with empty labels
        classifier.labels = []
        let image = createTestImage()
        
        let result = classifier.predictOnImage(image: image)
        
        XCTAssertEqual(result.names.count, 0)
        XCTAssertNil(result.probs)
    }
    
    func testPredictOnImageWithLargeImage() {
        // Test with a large image
        let largeImage = createTestImage(width: 4000, height: 3000)
        
        let result = classifier.predictOnImage(image: largeImage)
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result.orig_shape.width, 640) // Default size when no model
        XCTAssertEqual(result.orig_shape.height, 640)
    }
    
    // MARK: - Helper Methods
    
    private func createTestImage(width: CGFloat = 800, height: CGFloat = 600) -> CIImage {
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
        
        // Fill with a test pattern
        context.setFillColor(UIColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0).cgColor)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Add some colored rectangles
        context.setFillColor(UIColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0).cgColor)
        context.fill(CGRect(x: 100, y: 100, width: 200, height: 200))
        
        context.setFillColor(UIColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 1.0).cgColor)
        context.fill(CGRect(x: 400, y: 300, width: 150, height: 150))
        
        guard let cgImage = context.makeImage() else {
            return CIImage()
        }
        
        return CIImage(cgImage: cgImage)
    }
    
    private func createMockMLMultiArray(values: [Double]) -> MLMultiArray {
        let multiArray = try! MLMultiArray(shape: [values.count as NSNumber], dataType: .double)
        
        for (index, value) in values.enumerated() {
            multiArray[index] = NSNumber(value: value)
        }
        
        return multiArray
    }
}