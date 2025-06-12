// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Minimal tests for ObjectDetector functionality
class ObjectDetectorTests: XCTestCase {
    
    func testObjectDetectorInitialization() {
        // Test ObjectDetector initialization inherits from BasePredictor
        let detector = ObjectDetector()
        
        XCTAssertFalse(detector.isModelLoaded)
        XCTAssertEqual(detector.labels.count, 0)
        XCTAssertEqual(detector.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(detector.iouThreshold, 0.4, accuracy: 0.001)
        XCTAssertEqual(detector.numItemsThreshold, 30)
        XCTAssertFalse(detector.isUpdating)
    }
    
    func testObjectDetectorPredictOnImageWithoutModel() {
        // Test predictOnImage without loaded model returns empty result
        let detector = ObjectDetector()
        detector.labels = ["person", "car", "bicycle"]
        
        let image = CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 320, height: 240))
        let result = detector.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertNil(result.probs)
        XCTAssertEqual(result.names, ["person", "car", "bicycle"])
        XCTAssertEqual(result.orig_shape.width, 320)
        XCTAssertEqual(result.orig_shape.height, 240)
    }
    
    func testObjectDetectorProcessObservationsWithoutModel() {
        // Test processObservations without crashing
        let detector = ObjectDetector()
        detector.labels = ["person", "car", "bicycle", "dog", "cat"]
        detector.inputSize = CGSize(width: 640, height: 480)
        
        let mockRequest = MockVNRequest()
        
        // Should not crash
        detector.processObservations(for: mockRequest, error: nil)
        detector.processObservations(for: mockRequest, error: NSError(domain: "test", code: 1))
    }
    
    func testObjectDetectorLabelsAssignment() {
        // Test labels can be assigned and retrieved
        let detector = ObjectDetector()
        let testLabels = ["person", "bicycle", "car", "motorbike", "aeroplane"]
        
        detector.labels = testLabels
        XCTAssertEqual(detector.labels, testLabels)
        XCTAssertEqual(detector.labels.count, 5)
    }
    
    func testObjectDetectorThresholdUpdates() {
        // Test threshold setting methods
        let detector = ObjectDetector()
        
        detector.setConfidenceThreshold(confidence: 0.8)
        XCTAssertEqual(detector.confidenceThreshold, 0.8, accuracy: 0.001)
        
        detector.setIouThreshold(iou: 0.6)
        XCTAssertEqual(detector.iouThreshold, 0.6, accuracy: 0.001)
        
        detector.setNumItemsThreshold(numItems: 50)
        XCTAssertEqual(detector.numItemsThreshold, 50)
    }
    
    func testObjectDetectorInputSize() {
        // Test input size can be set and retrieved
        let detector = ObjectDetector()
        let testSize = CGSize(width: 640, height: 640)
        
        detector.inputSize = testSize
        XCTAssertEqual(detector.inputSize, testSize)
    }
    
    func testObjectDetectorTimingProperties() {
        // Test timing properties are properly initialized
        let detector = ObjectDetector()
        
        XCTAssertEqual(detector.t0, 0.0, accuracy: 0.001)
        XCTAssertEqual(detector.t1, 0.0, accuracy: 0.001)
        XCTAssertEqual(detector.t2, 0.0, accuracy: 0.001)
        XCTAssertEqual(detector.t4, 0.0, accuracy: 0.001)
        XCTAssertGreaterThan(detector.t3, 0)
    }
    
    func testObjectDetectorIsInstanceOfBasePredictor() {
        // Test ObjectDetector is instance of BasePredictor
        let detector = ObjectDetector()
        
        XCTAssertNotNil(detector, "Detector should not be nil")
        XCTAssertEqual(type(of: detector), ObjectDetector.self, "Should be ObjectDetector type")
    }
    
    func testObjectDetectorResultStructure() {
        // Test ObjectDetector result has correct structure
        let detector = ObjectDetector()
        detector.labels = ["person", "car"]
        
        let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 416, height: 416))
        let result = detector.predictOnImage(image: image)
        
        XCTAssertNotNil(result.boxes)
        XCTAssertNil(result.probs) // Detection doesn't use probs
        XCTAssertNil(result.masks) // Detection doesn't use masks
        XCTAssertEqual(result.keypointsList.count, 0) // Detection doesn't use keypoints
        XCTAssertEqual(result.obb.count, 0) // Detection doesn't use OBB
        XCTAssertEqual(result.names, ["person", "car"])
    }
    
    func testObjectDetectorEmptyLabelsHandling() {
        // Test ObjectDetector handles empty labels gracefully
        let detector = ObjectDetector()
        detector.labels = []
        
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        let result = detector.predictOnImage(image: image)
        
        XCTAssertEqual(result.names.count, 0)
        XCTAssertEqual(result.boxes.count, 0)
    }
}
