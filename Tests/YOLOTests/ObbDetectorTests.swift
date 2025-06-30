// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Comprehensive tests for ObbDetector functionality
class ObbDetectorTests: XCTestCase {
    
    var obbDetector: ObbDetector!
    
    override func setUp() {
        super.setUp()
        obbDetector = ObbDetector()
    }
    
    override func tearDown() {
        obbDetector = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testObbDetectorInitialization() {
        // Verify ObbDetector inherits proper initialization from BasePredictor
        XCTAssertFalse(obbDetector.isModelLoaded)
        XCTAssertEqual(obbDetector.labels.count, 0)
        XCTAssertEqual(obbDetector.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(obbDetector.iouThreshold, 0.4, accuracy: 0.001)
    }
    
    // MARK: - Process Observations Tests
    
    func testProcessObservationsWithEmptyResults() {
        // Test processing with no OBB results
        let request = MockVNRequestWithResults(results: [])
        
        // Should not crash
        obbDetector.processObservations(for: request, error: nil)
    }
    
    func testProcessObservationsWithValidOBBResults() {
        // Test processing with valid OBB detection outputs
        obbDetector.labels = ["vehicle", "ship", "plane", "storage-tank", "bridge"]
        obbDetector.inputSize = CGSize(width: 640, height: 480)
        
        // Create mock prediction data for OBB detection
        // OBB model output shape: [1, numFeatures, numAnchors]
        // numFeatures = 4 (cx, cy, w, h) + 1 (angle) + numClasses
        let numAnchors = 100
        let numClasses = 5
        let numFeatures = 4 + 1 + numClasses // 10 total
        let shape = [1, numFeatures, numAnchors] as [NSNumber]
        
        let predValues = createMockOBBPredictionValues(numAnchors: numAnchors, numClasses: numClasses)
        let predArray = createMockMLMultiArray(shape: shape, values: predValues)
        
        let observation = MockVNCoreMLFeatureValueObservation(multiArray: predArray)
        let request = MockVNRequestWithResults(results: [observation])
        
        let expectation = XCTestExpectation(description: "Process OBB observations")
        
        obbDetector.setOnResultsListener { result in
            // Verify result structure
            XCTAssertNotNil(result.obb)
            if let obbResults = result.obb {
                XCTAssertGreaterThanOrEqual(obbResults.count, 0)
            }
            expectation.fulfill()
        }
        
        obbDetector.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testProcessObservationsWithError() {
        // Test that errors are handled gracefully
        let request = MockVNRequestWithResults(results: [])
        let error = NSError(domain: "TestError", code: 100, userInfo: nil)
        
        // Should not crash
        obbDetector.processObservations(for: request, error: error)
    }
    
    // MARK: - Timing Tests
    
    func testTimingUpdate() {
        // Test that timing metrics are updated
        obbDetector.t1 = 0.05 // 50ms
        obbDetector.t2 = 0.0
        obbDetector.t3 = CACurrentMediaTime() - 0.033 // ~30 FPS
        obbDetector.t4 = 0.0
        
        let expectation = XCTestExpectation(description: "Timing update")
        
        obbDetector.setOnInferenceTimeListener { inferenceTime, fpsRate in
            XCTAssertGreaterThan(inferenceTime, 0)
            XCTAssertGreaterThan(fpsRate, 0)
            expectation.fulfill()
        }
        
        // Trigger timing update through observation processing
        let mockArray = createMockMLMultiArray(shape: [1, 10, 1], values: [0.0])
        let observation = MockVNCoreMLFeatureValueObservation(multiArray: mockArray)
        let request = MockVNRequestWithResults(results: [observation])
        obbDetector.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Predict on Image Tests
    
    func testPredictOnImageWithNoModel() {
        // Test prediction when no model is loaded
        let image = createTestImage()
        
        let result = obbDetector.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertNil(result.obb)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
    }
    
    func testPredictOnImageSetsInputSize() {
        // Test that predictOnImage properly sets input size
        let image = createTestImage(width: 1024, height: 768)
        
        _ = obbDetector.predictOnImage(image: image)
        
        XCTAssertEqual(obbDetector.inputSize.width, 1024)
        XCTAssertEqual(obbDetector.inputSize.height, 768)
    }
    
    // MARK: - Post Process OBB Tests
    
    func testPostProcessOBBWithNoDetections() {
        // Test post-processing with low confidence values
        let numAnchors = 10
        let numFeatures = 10 // 4 + 1 + 5 classes
        let shape = [1, numFeatures, numAnchors] as [NSNumber]
        
        // Create values with very low confidence
        var values = [Double]()
        for _ in 0..<(numFeatures * numAnchors) {
            values.append(Double.random(in: 0.0...0.1))
        }
        
        let multiArray = createMockMLMultiArray(shape: shape, values: values)
        
        let results = obbDetector.postProcessOBB(
            feature: multiArray,
            confidenceThreshold: 0.5,
            iouThreshold: 0.4
        )
        
        XCTAssertEqual(results.count, 0) // No detections should pass threshold
    }
    
    func testPostProcessOBBWithMultipleDetections() {
        // Test post-processing with multiple high-confidence detections
        let numAnchors = 5
        let numClasses = 5
        let numFeatures = 4 + 1 + numClasses
        let shape = [1, numFeatures, numAnchors] as [NSNumber]
        
        let values = createHighConfidenceOBBPredictionValues(numAnchors: numAnchors, numClasses: numClasses)
        let multiArray = createMockMLMultiArray(shape: shape, values: values)
        
        let results = obbDetector.postProcessOBB(
            feature: multiArray,
            confidenceThreshold: 0.3,
            iouThreshold: 0.4
        )
        
        XCTAssertGreaterThan(results.count, 0)
        
        // Verify result structure
        for result in results {
            XCTAssertGreaterThan(result.box.width, 0)
            XCTAssertGreaterThan(result.box.height, 0)
            XCTAssertGreaterThan(result.score, 0.3)
            XCTAssertGreaterThanOrEqual(result.box.angle, -Float.pi)
            XCTAssertLessThanOrEqual(result.box.angle, Float.pi)
            XCTAssertLessThan(result.cls, numClasses)
        }
    }
    
    // MARK: - Geometric Tests
    
    func testIOURotatedBoxes() {
        // Test IOU calculation for rotated boxes
        let box1 = RotatedBox(x: 100, y: 100, width: 50, height: 30, angle: 0)
        let box2 = RotatedBox(x: 100, y: 100, width: 50, height: 30, angle: 0)
        
        let iou = obbDetector.iouRotatedBoxes(boxA: box1, boxB: box2)
        XCTAssertEqual(iou, 1.0, accuracy: 0.001) // Identical boxes should have IOU = 1.0
        
        // Test non-overlapping boxes
        let box3 = RotatedBox(x: 200, y: 200, width: 50, height: 30, angle: 0)
        let iou2 = obbDetector.iouRotatedBoxes(boxA: box1, boxB: box3)
        XCTAssertEqual(iou2, 0.0, accuracy: 0.001) // Non-overlapping boxes should have IOU = 0.0
        
        // Test partially overlapping rotated boxes
        let box4 = RotatedBox(x: 120, y: 100, width: 50, height: 30, angle: Float.pi / 4)
        let iou3 = obbDetector.iouRotatedBoxes(boxA: box1, boxB: box4)
        XCTAssertGreaterThan(iou3, 0.0)
        XCTAssertLessThan(iou3, 1.0)
    }
    
    func testPolygonIntersection() {
        // Test Sutherland-Hodgman polygon intersection
        let square1 = [
            CGPoint(x: 0, y: 0),
            CGPoint(x: 10, y: 0),
            CGPoint(x: 10, y: 10),
            CGPoint(x: 0, y: 10)
        ]
        
        let square2 = [
            CGPoint(x: 5, y: 5),
            CGPoint(x: 15, y: 5),
            CGPoint(x: 15, y: 15),
            CGPoint(x: 5, y: 15)
        ]
        
        let intersection = obbDetector.sutherlandHodgman(subjectPolygon: square1, clipPolygon: square2)
        
        // The intersection should be a 5x5 square
        XCTAssertEqual(intersection.count, 4)
        let area = obbDetector.polygonArea(polygon: intersection)
        XCTAssertEqual(area, 25.0, accuracy: 0.001)
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndOBBDetection() {
        // Test complete OBB detection flow
        obbDetector.labels = ["vehicle", "ship", "plane", "storage-tank", "bridge"]
        obbDetector.setConfidenceThreshold(confidence: 0.4)
        obbDetector.setIouThreshold(iou: 0.5)
        
        let numAnchors = 20
        let numClasses = 5
        let numFeatures = 4 + 1 + numClasses
        let shape = [1, numFeatures, numAnchors] as [NSNumber]
        
        let predValues = createRealisticOBBPredictionValues(numAnchors: numAnchors, numClasses: numClasses)
        let predArray = createMockMLMultiArray(shape: shape, values: predValues)
        
        let observation = MockVNCoreMLFeatureValueObservation(multiArray: predArray)
        let request = MockVNRequestWithResults(results: [observation])
        
        let expectation = XCTestExpectation(description: "End to end OBB detection")
        
        obbDetector.setOnResultsListener { result in
            XCTAssertNotNil(result.obb)
            if let obbResults = result.obb {
                XCTAssertGreaterThan(obbResults.count, 0)
                
                // Verify first detection
                if let firstOBB = obbResults.first {
                    XCTAssertGreaterThan(firstOBB.confidence, 0.4)
                    XCTAssertNotNil(firstOBB.cls)
                    XCTAssertGreaterThan(firstOBB.box.width, 0)
                    XCTAssertGreaterThan(firstOBB.box.height, 0)
                    
                    // Verify class name is valid
                    let validClasses = ["vehicle", "ship", "plane", "storage-tank", "bridge"]
                    XCTAssertTrue(validClasses.contains(firstOBB.cls))
                }
            }
            
            expectation.fulfill()
        }
        
        obbDetector.processObservations(for: request, error: nil)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Helper Methods
    
    private func createTestImage(width: CGFloat = 640, height: CGFloat = 480) -> CIImage {
        return CIImage(color: CIColor(red: 0.0, green: 1.0, blue: 1.0)).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
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
    
    private func createMockOBBPredictionValues(numAnchors: Int, numClasses: Int) -> [Double] {
        var values = [Double]()
        
        for anchor in 0..<numAnchors {
            // Box center and dimensions (cx, cy, w, h)
            values.append(Double(320 + anchor * 10)) // cx
            values.append(Double(240 + anchor * 10)) // cy
            values.append(Double(60 + anchor * 5))   // width
            values.append(Double(40 + anchor * 5))   // height
            
            // Rotation angle
            values.append(Double.random(in: -Double.pi...Double.pi))
            
            // Class probabilities
            for cls in 0..<numClasses {
                if anchor % numClasses == cls {
                    values.append(0.7) // High confidence for one class
                } else {
                    values.append(0.1)
                }
            }
        }
        
        return values
    }
    
    private func createHighConfidenceOBBPredictionValues(numAnchors: Int, numClasses: Int) -> [Double] {
        var values = [Double]()
        
        for anchor in 0..<numAnchors {
            // Box parameters
            values.append(Double(100 + anchor * 120)) // cx
            values.append(Double(100 + anchor * 80))  // cy
            values.append(Double(80))  // width
            values.append(Double(50))  // height
            values.append(Double(anchor) * 0.3) // angle
            
            // High confidence for one class
            for cls in 0..<numClasses {
                if cls == anchor % numClasses {
                    values.append(0.85)
                } else {
                    values.append(0.05)
                }
            }
        }
        
        return values
    }
    
    private func createRealisticOBBPredictionValues(numAnchors: Int, numClasses: Int) -> [Double] {
        var values = [Double]()
        
        for anchor in 0..<numAnchors {
            // Realistic oriented box parameters
            let cx = Double.random(in: 100...540)
            let cy = Double.random(in: 100...380)
            let w = Double.random(in: 40...120)
            let h = Double.random(in: 30...80)
            let angle = Double.random(in: -1.0...1.0) // radians
            
            values.append(cx)
            values.append(cy)
            values.append(w)
            values.append(h)
            values.append(angle)
            
            // Class probabilities (high for first few anchors)
            let isHighConfidence = anchor < 5
            for cls in 0..<numClasses {
                if isHighConfidence && cls == anchor % numClasses {
                    values.append(Double.random(in: 0.7...0.95))
                } else {
                    values.append(Double.random(in: 0.01...0.2))
                }
            }
        }
        
        return values
    }
}