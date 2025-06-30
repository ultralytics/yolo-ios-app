// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Comprehensive tests for Classifier functionality
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
        // Verify Classifier inherits proper initialization from BasePredictor
        XCTAssertFalse(classifier.isModelLoaded)
        XCTAssertEqual(classifier.labels.count, 0)
        XCTAssertEqual(classifier.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(classifier.iouThreshold, 0.4, accuracy: 0.001)
    }
    
    // MARK: - Threshold Configuration Tests
    
    func testSetConfidenceThreshold() {
        // Test confidence threshold setting
        classifier.setConfidenceThreshold(confidence: 0.7)
        XCTAssertEqual(classifier.confidenceThreshold, 0.7, accuracy: 0.001)
        
        // Test boundary values
        classifier.setConfidenceThreshold(confidence: 0.0)
        XCTAssertEqual(classifier.confidenceThreshold, 0.0, accuracy: 0.001)
        
        classifier.setConfidenceThreshold(confidence: 1.0)
        XCTAssertEqual(classifier.confidenceThreshold, 1.0, accuracy: 0.001)
    }
    
    func testSetIouThreshold() {
        // Test IoU threshold setting (though not typically used in classification)
        classifier.setIouThreshold(iou: 0.5)
        XCTAssertEqual(classifier.iouThreshold, 0.5, accuracy: 0.001)
    }
    
    // MARK: - Process Observations Tests with MLMultiArray
    
    func testProcessObservationsWithMLMultiArray() {
        // Test processing with MLMultiArray results
        classifier.labels = ["cat", "dog", "bird", "fish", "horse"]
        classifier.inputSize = CGSize(width: 224, height: 224)
        
        let mockMultiArray = createMockMLMultiArray(values: [0.1, 0.7, 0.05, 0.1, 0.05])
        let observation = MockVNCoreMLFeatureValueObservation(multiArray: mockMultiArray)
        let request = MockVNRequestWithResults(results: [observation])
        
        let expectation = XCTestExpectation(description: "Process MLMultiArray observations")
        
        let mockListener = MockResultsListener()
        mockListener.onResultHandler = { result in
            XCTAssertNotNil(result.probs)
            
            // Check top1 prediction
            XCTAssertEqual(result.probs?.top1, "dog")
            XCTAssertEqual(result.probs?.top1Conf ?? 0, 0.7, accuracy: 0.001)
            
            // Check top5 predictions
            XCTAssertEqual(result.probs?.top5.count, 5)
            XCTAssertEqual(result.probs?.top5[0], "dog")
            XCTAssertEqual(result.probs?.top5[1], "cat")
            XCTAssertEqual(result.probs?.top5[2], "fish")
            
            // Check confidence values are sorted in descending order
            let confs = result.probs?.top5Confs ?? []
            for i in 0..<confs.count-1 {
                XCTAssertGreaterThanOrEqual(confs[i], confs[i+1])
            }
            
            expectation.fulfill()
        }
        classifier.currentOnResultsListener = mockListener
        
        classifier.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testProcessObservationsWithVNClassificationObservations() {
        // Test processing with VNClassificationObservation results
        classifier.labels = ["cat", "dog", "bird", "fish", "horse"]
        classifier.inputSize = CGSize(width: 224, height: 224)
        
        let mockObservations = [
            MockVNClassificationObservation(identifier: "dog", confidence: 0.85),
            MockVNClassificationObservation(identifier: "cat", confidence: 0.10),
            MockVNClassificationObservation(identifier: "wolf", confidence: 0.03),
            MockVNClassificationObservation(identifier: "fox", confidence: 0.01),
            MockVNClassificationObservation(identifier: "coyote", confidence: 0.01)
        ]
        
        let request = MockVNRequestWithResults(results: mockObservations)
        
        let expectation = XCTestExpectation(description: "Process VNClassification observations")
        
        let mockListener = MockResultsListener()
        mockListener.onResultHandler = { result in
            XCTAssertNotNil(result.probs)
            
            // Check top1 prediction
            XCTAssertEqual(result.probs?.top1, "dog")
            XCTAssertEqual(result.probs?.top1Conf ?? 0, 0.85, accuracy: 0.001)
            
            // Check top5 predictions
            XCTAssertEqual(result.probs?.top5.count, 5)
            XCTAssertEqual(result.probs?.top5[0], "dog")
            XCTAssertEqual(result.probs?.top5[1], "cat")
            XCTAssertEqual(result.probs?.top5[4], "coyote")
            
            expectation.fulfill()
        }
        classifier.currentOnResultsListener = mockListener
        
        classifier.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testProcessObservationsWithFewerThan5Classes() {
        // Test handling when there are fewer than 5 classification results
        classifier.labels = ["cat", "dog", "bird"]
        
        let mockObservations = [
            MockVNClassificationObservation(identifier: "dog", confidence: 0.8),
            MockVNClassificationObservation(identifier: "cat", confidence: 0.15),
            MockVNClassificationObservation(identifier: "bird", confidence: 0.05)
        ]
        
        let request = MockVNRequestWithResults(results: mockObservations)
        
        let expectation = XCTestExpectation(description: "Process fewer than 5 observations")
        
        let mockListener = MockResultsListener()
        mockListener.onResultHandler = { result in
            XCTAssertNotNil(result.probs)
            XCTAssertEqual(result.probs?.top5.count, 3) // Only 3 classes available
            XCTAssertEqual(result.probs?.top5Confs.count, 3)
            expectation.fulfill()
        }
        
        classifier.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testProcessObservationsWithEmptyResults() {
        // Test handling of empty results
        let request = MockVNRequestWithResults(results: [])
        
        let expectation = XCTestExpectation(description: "Process empty observations")
        
        let mockListener = MockResultsListener()
        mockListener.onResultHandler = { result in
            // Should still receive a result with empty probs
            XCTAssertNotNil(result)
            XCTAssertEqual(result.probs?.top1, "")
            XCTAssertEqual(result.probs?.top1Conf, 0)
            XCTAssertEqual(result.probs?.top5.count, 0)
            expectation.fulfill()
        }
        classifier.currentOnResultsListener = mockListener
        
        classifier.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testProcessObservationsPerformanceMetrics() {
        // Test that performance metrics are calculated
        let request = MockVNRequestWithResults(results: [])
        
        let inferenceExpectation = XCTestExpectation(description: "Inference time callback")
        
        let mockTimeListener = MockInferenceTimeListener()
        mockTimeListener.onInferenceTimeHandler = { inferenceTime, fpsRate in
            XCTAssertGreaterThanOrEqual(inferenceTime, 0)
            XCTAssertGreaterThan(fpsRate, 0)
            inferenceExpectation.fulfill()
        }
        classifier.currentOnInferenceTimeListener = mockTimeListener
        
        classifier.processObservations(for: request, error: nil)
        
        wait(for: [inferenceExpectation], timeout: 1.0)
    }
    
    // MARK: - Predict on Image Tests
    
    func testPredictOnImageWithNoModel() {
        // Test prediction when no model is loaded
        let image = createTestImage()
        
        let result = classifier.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
        XCTAssertEqual(result.orig_shape, classifier.inputSize)
    }
    
    func testPredictOnImageWithMLMultiArray() {
        // Test that predictOnImage returns default result when no model is loaded
        classifier.labels = ["apple", "banana", "orange", "grape", "strawberry"]
        
        let image = createTestImage()
        let result = classifier.predictOnImage(image: image)
        
        // Without a loaded model, result should have default values
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
        XCTAssertEqual(result.names, classifier.labels)
        
        // For actual prediction testing, we simulate via processObservations
        let mockMultiArray = createMockMLMultiArray(values: [0.05, 0.1, 0.6, 0.2, 0.05])
        let observation = MockVNCoreMLFeatureValueObservation(multiArray: mockMultiArray)
        let mockRequest = MockVNRequestWithResults(results: [observation])
        
        classifier.inputSize = CGSize(width: 224, height: 224)
        classifier.processObservations(for: mockRequest, error: nil)
    }
    
    func testPredictOnImageWithVNClassifications() {
        // Test that predictOnImage returns default result when no model is loaded
        classifier.labels = ["cat", "dog", "bird"]
        
        let image = createTestImage()
        let result = classifier.predictOnImage(image: image)
        
        // Without a loaded model, result should have default values
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
        XCTAssertEqual(result.names, classifier.labels)
        
        // For actual prediction testing, we simulate via processObservations
        let mockObservations = [
            MockVNClassificationObservation(identifier: "cat", confidence: 0.95),
            MockVNClassificationObservation(identifier: "dog", confidence: 0.04),
            MockVNClassificationObservation(identifier: "bird", confidence: 0.01)
        ]
        let mockRequest = MockVNRequestWithResults(results: mockObservations)
        
        classifier.inputSize = CGSize(width: 224, height: 224)
        classifier.processObservations(for: mockRequest, error: nil)
    }
    
    func testPredictOnImageSetsInputSize() {
        // Test that predictOnImage properly sets input size
        let image = createTestImage(width: 299, height: 299)
        
        _ = classifier.predictOnImage(image: image)
        
        XCTAssertEqual(classifier.inputSize.width, 299)
        XCTAssertEqual(classifier.inputSize.height, 299)
    }
    
    // MARK: - Error Handling Tests
    
    func testProcessObservationsWithError() {
        // Test that errors are handled gracefully
        let request = MockVNRequestWithResults(results: [])
        let error = NSError(domain: "TestError", code: 100, userInfo: nil)
        
        // Should not crash
        classifier.processObservations(for: request, error: error)
    }
    
    func testProcessObservationsWithWrongResultType() {
        // Test handling of unexpected result types
        let request = MockVNRequestWithResults(results: ["Not a valid observation type"])
        
        // Should not crash and should return empty probs
        let expectation = XCTestExpectation(description: "Handle wrong result type")
        
        let mockListener = MockResultsListener()
        mockListener.onResultHandler = { result in
            XCTAssertEqual(result.probs?.top1, "")
            XCTAssertEqual(result.probs?.top5.count, 0)
            expectation.fulfill()
        }
        classifier.currentOnResultsListener = mockListener
        
        classifier.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndClassification() {
        // Test complete classification flow
        classifier.labels = Array(0..<1000).map { "class_\($0)" } // ImageNet-style labels
        
        // Create mock results with realistic confidence distribution
        var values = [Double](repeating: 0.0001, count: 1000)
        values[42] = 0.92  // High confidence for one class
        values[100] = 0.04
        values[200] = 0.02
        values[300] = 0.01
        values[400] = 0.005
        
        let mockMultiArray = createMockMLMultiArray(values: values)
        let observation = MockVNCoreMLFeatureValueObservation(multiArray: mockMultiArray)
        let request = MockVNRequestWithResults(results: [observation])
        
        let expectation = XCTestExpectation(description: "End to end classification")
        
        let mockListener = MockResultsListener()
        mockListener.onResultHandler = { result in
            XCTAssertNotNil(result.probs)
            XCTAssertEqual(result.probs?.top1, "class_42")
            XCTAssertEqual(result.probs?.top1Conf ?? 0, 0.92, accuracy: 0.001)
            
            // Verify top5 contains the highest confidence classes
            let top5 = result.probs?.top5 ?? []
            XCTAssertTrue(top5.contains(where: { $0 == "class_42" }))
            XCTAssertTrue(top5.contains(where: { $0 == "class_100" }))
            XCTAssertTrue(top5.contains(where: { $0 == "class_200" }))
            
            if let fps = result.fps {
                XCTAssertGreaterThan(fps, 0)
            }
            XCTAssertGreaterThanOrEqual(result.speed, 0)
            
            expectation.fulfill()
        }
        classifier.currentOnResultsListener = mockListener
        
        classifier.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testClassificationWithSingleClass() {
        // Test edge case with single class
        classifier.labels = ["only_class"]
        
        let mockMultiArray = createMockMLMultiArray(values: [1.0])
        let observation = MockVNCoreMLFeatureValueObservation(multiArray: mockMultiArray)
        let request = MockVNRequestWithResults(results: [observation])
        
        let expectation = XCTestExpectation(description: "Single class classification")
        
        let mockListener = MockResultsListener()
        mockListener.onResultHandler = { result in
            XCTAssertEqual(result.probs?.top1, "only_class")
            XCTAssertEqual(result.probs?.top1Conf ?? 0, 1.0, accuracy: 0.001)
            XCTAssertEqual(result.probs?.top5.count, 1)
            expectation.fulfill()
        }
        classifier.currentOnResultsListener = mockListener
        
        classifier.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Helper Methods
    
    private func createTestImage(width: CGFloat = 224, height: CGFloat = 224) -> CIImage {
        return CIImage(color: CIColor(red: 0.0, green: 1.0, blue: 0.0)).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }
    
    private func createMockMLMultiArray(values: [Double]) -> MLMultiArray {
        let shape = [NSNumber(value: values.count)]
        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .double) else {
            fatalError("Failed to create MLMultiArray")
        }
        
        for (index, value) in values.enumerated() {
            multiArray[index] = NSNumber(value: value)
        }
        
        return multiArray
    }
}