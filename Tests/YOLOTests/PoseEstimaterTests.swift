// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Minimal tests for PoseEstimater functionality
class PoseEstimaterTests: XCTestCase {
    
    func testPoseEstimaterInitialization() {
        // Test PoseEstimater initialization inherits from BasePredictor
        let poseEstimater = PoseEstimater()
        
        XCTAssertFalse(poseEstimater.isModelLoaded)
        XCTAssertEqual(poseEstimater.labels.count, 0)
        XCTAssertEqual(poseEstimater.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(poseEstimater.iouThreshold, 0.4, accuracy: 0.001)
        XCTAssertEqual(poseEstimater.numItemsThreshold, 30)
        XCTAssertFalse(poseEstimater.isUpdating)
    }
    
    func testPoseEstimaterPredictOnImageWithoutModel() {
        // Test predictOnImage without loaded model returns empty result
        let poseEstimater = PoseEstimater()
        poseEstimater.labels = ["person"]
        
        let image = CIImage(color: .yellow).cropped(to: CGRect(x: 0, y: 0, width: 640, height: 640))
        let result = poseEstimater.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertNil(result.probs)
        XCTAssertNil(result.masks)
        XCTAssertEqual(result.keypointsList.count, 0)
        XCTAssertEqual(result.names, ["person"])
        XCTAssertEqual(result.orig_shape.width, 640)
        XCTAssertEqual(result.orig_shape.height, 640)
    }
    
    func testPoseEstimaterProcessObservationsWithoutModel() {
        // Test processObservations without crashing
        let poseEstimater = PoseEstimater()
        poseEstimater.labels = ["person"]
        poseEstimater.inputSize = CGSize(width: 640, height: 480)
        
        let mockRequest = MockVNRequest()
        
        // Should not crash
        poseEstimater.processObservations(for: mockRequest, error: nil)
        poseEstimater.processObservations(for: mockRequest, error: NSError(domain: "test", code: 1))
    }
    
    func testPoseEstimaterLabelsAssignment() {
        // Test labels can be assigned and retrieved
        let poseEstimater = PoseEstimater()
        let testLabels = ["person"]
        
        poseEstimater.labels = testLabels
        XCTAssertEqual(poseEstimater.labels, testLabels)
        XCTAssertEqual(poseEstimater.labels.count, 1)
    }
    
    func testPoseEstimaterInputSize() {
        // Test input size can be set and retrieved
        let poseEstimater = PoseEstimater()
        let testSize = CGSize(width: 640, height: 480)
        
        poseEstimater.inputSize = testSize
        XCTAssertEqual(poseEstimater.inputSize, testSize)
    }
    
    func testPoseEstimaterTimingProperties() {
        // Test timing properties are properly initialized
        let poseEstimater = PoseEstimater()
        
        XCTAssertEqual(poseEstimater.t0, 0.0, accuracy: 0.001)
        XCTAssertEqual(poseEstimater.t1, 0.0, accuracy: 0.001)
        XCTAssertEqual(poseEstimater.t2, 0.0, accuracy: 0.001)
        XCTAssertEqual(poseEstimater.t4, 0.0, accuracy: 0.001)
        XCTAssertGreaterThan(poseEstimater.t3, 0)
    }
    
    func testPoseEstimaterIsInstanceOfBasePredictor() {
        // Test PoseEstimater is instance of BasePredictor
        let poseEstimater = PoseEstimater()
        
        XCTAssertTrue(poseEstimater is BasePredictor)
        XCTAssertTrue(poseEstimater is Predictor)
    }
    
    func testPoseEstimaterResultStructure() {
        // Test PoseEstimater result has correct structure
        let poseEstimater = PoseEstimater()
        poseEstimater.labels = ["person"]
        
        let image = CIImage(color: .purple).cropped(to: CGRect(x: 0, y: 0, width: 416, height: 416))
        let result = poseEstimater.predictOnImage(image: image)
        
        XCTAssertNotNil(result.boxes)
        XCTAssertNil(result.probs) // Pose doesn't use probs
        XCTAssertNil(result.masks) // Pose doesn't use masks
        XCTAssertNotNil(result.keypointsList) // Pose uses keypoints
        XCTAssertEqual(result.obb.count, 0) // Pose doesn't use OBB
        XCTAssertEqual(result.names, ["person"])
    }
    
    func testPoseEstimaterColorsForMaskProperty() {
        // Test colorsForMask property exists and can be modified
        let poseEstimater = PoseEstimater()
        
        XCTAssertEqual(poseEstimater.colorsForMask.count, 0)
        
        poseEstimater.colorsForMask = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        XCTAssertEqual(poseEstimater.colorsForMask.count, 3)
        XCTAssertEqual(poseEstimater.colorsForMask[0].red, 255)
        XCTAssertEqual(poseEstimater.colorsForMask[1].green, 255)
        XCTAssertEqual(poseEstimater.colorsForMask[2].blue, 255)
    }
    
    func testPoseEstimaterModelInputSize() {
        // Test model input size properties
        let poseEstimater = PoseEstimater()
        
        XCTAssertEqual(poseEstimater.modelInputSize.width, 0)
        XCTAssertEqual(poseEstimater.modelInputSize.height, 0)
    }
}
