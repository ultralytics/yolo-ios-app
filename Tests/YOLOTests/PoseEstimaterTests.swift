// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Comprehensive tests for PoseEstimater functionality
class PoseEstimaterTests: XCTestCase {
    
    var poseEstimater: PoseEstimater!
    
    override func setUp() {
        super.setUp()
        poseEstimater = PoseEstimater()
    }
    
    override func tearDown() {
        poseEstimater = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testPoseEstimaterInitialization() {
        // Verify PoseEstimater inherits proper initialization from BasePredictor
        XCTAssertFalse(poseEstimater.isModelLoaded)
        XCTAssertEqual(poseEstimater.labels.count, 0)
        XCTAssertEqual(poseEstimater.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(poseEstimater.iouThreshold, 0.4, accuracy: 0.001)
        XCTAssertEqual(poseEstimater.colorsForMask.count, 0)
    }
    
    // MARK: - Process Observations Tests
    
    func testProcessObservationsWithEmptyResults() {
        // Test processing with no pose results
        let request = MockVNRequestWithResults(results: [])
        
        // Should not crash
        poseEstimater.processObservations(for: request, error: nil)
    }
    
    func testProcessObservationsWithValidPoseResults() {
        // Skip this test as it requires mocking VNCoreMLFeatureValueObservation
        XCTSkip("This test requires a real CoreML model and VNCoreMLFeatureValueObservation")
    }
    
    func testProcessObservationsWithError() {
        // Test that errors are handled gracefully
        let request = MockVNRequestWithResults(results: [])
        let error = NSError(domain: "TestError", code: 100, userInfo: nil)
        
        // Should not crash
        poseEstimater.processObservations(for: request, error: error)
    }
    
    // MARK: - Timing Tests
    
    func testTimingUpdate() {
        // Test that timing metrics are updated
        poseEstimater.t1 = 0.05 // 50ms
        poseEstimater.t2 = 0.0
        poseEstimater.t3 = CACurrentMediaTime() - 0.033 // ~30 FPS
        poseEstimater.t4 = 0.0
        
        let expectation = XCTestExpectation(description: "Timing update")
        
        // Set up a mock inference time listener
        let mockListener = MockInferenceTimeListener()
        mockListener.onInferenceTimeHandler = { inferenceTime, fpsRate in
            XCTAssertGreaterThan(inferenceTime, 0)
            XCTAssertGreaterThan(fpsRate, 0)
            expectation.fulfill()
        }
        poseEstimater.currentOnInferenceTimeListener = mockListener
        
        // Trigger timing update through observation processing
        let request = MockVNRequestWithResults(results: [])
        poseEstimater.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    // MARK: - Predict on Image Tests
    
    func testPredictOnImageWithNoModel() {
        // Test prediction when no model is loaded
        let image = createTestImage()
        
        let result = poseEstimater.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertTrue(result.keypointsList.isEmpty)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
    }
    
    func testPredictOnImageSetsInputSize() {
        // Test that predictOnImage properly sets input size
        let image = createTestImage(width: 800, height: 600)
        
        _ = poseEstimater.predictOnImage(image: image)
        
        XCTAssertEqual(poseEstimater.inputSize.width, 800)
        XCTAssertEqual(poseEstimater.inputSize.height, 600)
    }
    
    // MARK: - Post Process Pose Tests
    
    func testPostProcessPoseWithNoDetections() {
        // Test post-processing with low confidence values
        let numAnchors = 10
        let outputFeatures = 56
        let shape = [1, outputFeatures, numAnchors] as [NSNumber]
        
        // Create values with very low confidence
        var values = [Double]()
        for _ in 0..<(outputFeatures * numAnchors) {
            values.append(Double.random(in: 0.0...0.1))
        }
        
        let multiArray = createMockMLMultiArray(shape: shape, values: values)
        
        let results = poseEstimater.PostProcessPose(
            prediction: multiArray,
            confidenceThreshold: 0.5,
            iouThreshold: 0.4
        )
        
        XCTAssertEqual(results.count, 0) // No detections should pass threshold
    }
    
    func testPostProcessPoseWithMultiplePersons() {
        // Test post-processing with multiple high-confidence person detections
        let numAnchors = 5
        let outputFeatures = 56
        let shape = [1, outputFeatures, numAnchors] as [NSNumber]
        
        let values = createHighConfidencePosePredictionValues(numAnchors: numAnchors)
        let multiArray = createMockMLMultiArray(shape: shape, values: values)
        
        let results = poseEstimater.PostProcessPose(
            prediction: multiArray,
            confidenceThreshold: 0.3,
            iouThreshold: 0.4
        )
        
        XCTAssertGreaterThan(results.count, 0)
        
        // Verify result structure
        for result in results {
            XCTAssertGreaterThan(result.box.xywh.width, 0)
            XCTAssertGreaterThan(result.box.xywh.height, 0)
            XCTAssertGreaterThan(result.box.conf, 0.3)
            
            // Verify keypoints
            XCTAssertEqual(result.keypoints.xyn.count, 17) // 17 keypoints for human pose
            XCTAssertEqual(result.keypoints.xy.count, 17)
            XCTAssertEqual(result.keypoints.conf.count, 17)
        }
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndPoseEstimation() {
        // Skip this test as it requires mocking VNCoreMLFeatureValueObservation
        XCTSkip("This test requires a real CoreML model and VNCoreMLFeatureValueObservation")
    }
    
    // MARK: - Helper Methods
    
    private func createTestImage(width: CGFloat = 640, height: CGFloat = 480) -> CIImage {
        return CIImage(color: CIColor(red: 1.0, green: 0.5, blue: 0.0)).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }
    
    private func createMockMLMultiArray(shape: [NSNumber], values: [Double]) -> MLMultiArray {
        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            fatalError("Failed to create MLMultiArray")
        }
        
        for (index, value) in values.enumerated() {
            if index < multiArray.count {
                multiArray[index] = NSNumber(value: Float(value))
            }
        }
        
        return multiArray
    }
    
    private func createMockPosePredictionValues(numAnchors: Int) -> [Double] {
        var values = [Double]()
        let outputFeatures = 56 // 4 (box) + 1 (objectness) + 51 (17 keypoints * 3)
        
        for anchor in 0..<numAnchors {
            // Box coordinates (x, y, w, h)
            values.append(Double(320 + anchor * 10)) // x
            values.append(Double(240 + anchor * 10)) // y
            values.append(Double(80 + anchor * 5))   // width
            values.append(Double(80 + anchor * 5))   // height
            
            // Objectness score
            values.append(Double.random(in: 0.1...0.5))
            
            // 17 keypoints * 3 (x, y, confidence)
            for _ in 0..<17 {
                values.append(Double.random(in: 100...500)) // x
                values.append(Double.random(in: 100...400)) // y
                values.append(Double.random(in: 0.3...0.9)) // confidence
            }
        }
        
        return values
    }
    
    private func createHighConfidencePosePredictionValues(numAnchors: Int) -> [Double] {
        var values = [Double]()
        
        for anchor in 0..<numAnchors {
            // Box coordinates
            values.append(Double(100 + anchor * 150)) // x
            values.append(Double(100 + anchor * 100)) // y
            values.append(Double(120)) // width
            values.append(Double(200)) // height
            
            // High objectness score
            values.append(0.85)
            
            // 17 keypoints with high confidence
            for i in 0..<17 {
                values.append(Double(150 + i * 10)) // x
                values.append(Double(100 + i * 15)) // y
                values.append(0.8) // high confidence
            }
        }
        
        return values
    }
    
    private func createRealisticPosePredictionValues(numAnchors: Int) -> [Double] {
        var values = [Double]()
        
        for anchor in 0..<numAnchors {
            // Realistic box coordinates for a person
            let x = Double.random(in: 200...440)
            let y = Double.random(in: 100...300)
            let w = Double.random(in: 80...150)
            let h = Double.random(in: 150...250)
            
            values.append(x)
            values.append(y)
            values.append(w)
            values.append(h)
            
            // Objectness score (high for first few anchors)
            let objectness = anchor < 3 ? Double.random(in: 0.7...0.95) : Double.random(in: 0.1...0.3)
            values.append(objectness)
            
            // 17 keypoints in anatomically plausible positions
            let keypointPositions = [
                (x, y - h * 0.4),      // nose
                (x - w * 0.1, y - h * 0.35),  // left eye
                (x + w * 0.1, y - h * 0.35),  // right eye
                (x - w * 0.15, y - h * 0.3),  // left ear
                (x + w * 0.15, y - h * 0.3),  // right ear
                (x - w * 0.3, y - h * 0.1),   // left shoulder
                (x + w * 0.3, y - h * 0.1),   // right shoulder
                (x - w * 0.35, y + h * 0.1),  // left elbow
                (x + w * 0.35, y + h * 0.1),  // right elbow
                (x - w * 0.4, y + h * 0.3),   // left wrist
                (x + w * 0.4, y + h * 0.3),   // right wrist
                (x - w * 0.2, y + h * 0.3),   // left hip
                (x + w * 0.2, y + h * 0.3),   // right hip
                (x - w * 0.2, y + h * 0.5),   // left knee
                (x + w * 0.2, y + h * 0.5),   // right knee
                (x - w * 0.2, y + h * 0.7),   // left ankle
                (x + w * 0.2, y + h * 0.7)    // right ankle
            ]
            
            for (kx, ky) in keypointPositions {
                values.append(kx)
                values.append(ky)
                values.append(objectness > 0.5 ? Double.random(in: 0.6...0.95) : Double.random(in: 0.1...0.4))
            }
        }
        
        return values
    }
}