// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Minimal tests for Classifier functionality
class ClassifierTests: XCTestCase {
    
    func testClassifierInitialization() {
        // Test Classifier initialization inherits from BasePredictor
        let classifier = Classifier()
        
        XCTAssertFalse(classifier.isModelLoaded)
        XCTAssertEqual(classifier.labels.count, 0)
        XCTAssertEqual(classifier.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(classifier.iouThreshold, 0.4, accuracy: 0.001)
        XCTAssertFalse(classifier.isUpdating)
    }
    
    func testClassifierPredictOnImageWithoutModel() {
        // Test predictOnImage without loaded model returns empty result
        let classifier = Classifier()
        classifier.labels = ["cat", "dog", "bird"]
        
        let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        let result = classifier.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertNotNil(result.probs)
        XCTAssertEqual(result.probs?.top1, "")
        XCTAssertEqual(result.probs?.top5.count, 0)
        XCTAssertEqual(result.names, ["cat", "dog", "bird"])
    }
    
    func testClassifierProcessObservationsWithoutModel() {
        // Test processObservations without crashing
        let classifier = Classifier()
        classifier.labels = ["person", "car", "bicycle"]
        classifier.inputSize = CGSize(width: 640, height: 480)
        
        let mockRequest = MockVNRequest()
        
        // Should not crash
        classifier.processObservations(for: mockRequest, error: nil)
        classifier.processObservations(for: mockRequest, error: NSError(domain: "test", code: 1))
    }
    
    func testClassifierLabelsAssignment() {
        // Test labels can be assigned and retrieved
        let classifier = Classifier()
        let testLabels = ["cat", "dog", "bird", "fish", "mouse"]
        
        classifier.labels = testLabels
        XCTAssertEqual(classifier.labels, testLabels)
        XCTAssertEqual(classifier.labels.count, 5)
    }
    
    func testClassifierInputSize() {
        // Test input size can be set and retrieved
        let classifier = Classifier()
        let testSize = CGSize(width: 224, height: 224)
        
        classifier.inputSize = testSize
        XCTAssertEqual(classifier.inputSize, testSize)
    }
    
    func testClassifierTimingProperties() {
        // Test timing properties are properly initialized
        let classifier = Classifier()
        
        XCTAssertEqual(classifier.t0, 0.0, accuracy: 0.001)
        XCTAssertEqual(classifier.t1, 0.0, accuracy: 0.001)
        XCTAssertEqual(classifier.t2, 0.0, accuracy: 0.001)
        XCTAssertEqual(classifier.t4, 0.0, accuracy: 0.001)
        XCTAssertGreaterThan(classifier.t3, 0)
    }
    
    func testClassifierIsInstanceOfBasePredictor() {
        // Test Classifier is instance of BasePredictor
        let classifier = Classifier()
        
        XCTAssertTrue(classifier is BasePredictor)
        XCTAssertTrue(classifier is Predictor)
    }
    
    func testClassifierOverridesBaseMethods() {
        // Test Classifier overrides base methods
        let classifier = Classifier()
        
        // Test that methods can be called without crashing
        classifier.setConfidenceThreshold(confidence: 0.5)
        classifier.setIouThreshold(iou: 0.6)
        
        XCTAssertEqual(classifier.confidenceThreshold, 0.5, accuracy: 0.001)
        XCTAssertEqual(classifier.iouThreshold, 0.6, accuracy: 0.001)
    }
    
    func testClassifierEmptyLabelsHandling() {
        // Test Classifier handles empty labels gracefully
        let classifier = Classifier()
        classifier.labels = []
        
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        let result = classifier.predictOnImage(image: image)
        
        XCTAssertEqual(result.names.count, 0)
        XCTAssertNotNil(result.probs)
    }
}
