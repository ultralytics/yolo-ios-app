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
        // Skip this test as it requires mocking VNCoreMLFeatureValueObservation which is not possible
        XCTSkip("This test requires a real CoreML model and VNCoreMLFeatureValueObservation")
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
        // Skip this test as it requires mocking VNCoreMLFeatureValueObservation
        XCTSkip("This test requires a real CoreML model and VNCoreMLFeatureValueObservation")
    }
    
    // MARK: - Predict on Image Tests
    
    func testPredictOnImageWithNoModel() {
        // Test prediction when no model is loaded
        let image = createTestImage()
        
        let result = obbDetector.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertEqual(result.obb.count, 0)
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
            XCTAssertGreaterThan(result.box.w, 0)
            XCTAssertGreaterThan(result.box.h, 0)
            XCTAssertGreaterThan(result.score, 0.3)
            XCTAssertGreaterThanOrEqual(result.box.angle, -Float.pi)
            XCTAssertLessThanOrEqual(result.box.angle, Float.pi)
            XCTAssertLessThan(result.cls, numClasses)
        }
    }
    
    // MARK: - Geometric Tests
    
    func testIOURotatedBoxes() {
        // Test IOU calculation for rotated boxes
        let box1 = OBB(cx: 100, cy: 100, w: 50, h: 30, angle: 0)
        let box2 = OBB(cx: 100, cy: 100, w: 50, h: 30, angle: 0)
        
        let iou = obbIoU(box1, box2)
        XCTAssertEqual(iou, 1.0, accuracy: 0.001) // Identical boxes should have IOU = 1.0
        
        // Test non-overlapping boxes
        let box3 = OBB(cx: 200, cy: 200, w: 50, h: 30, angle: 0)
        let iou2 = obbIoU(box1, box3)
        XCTAssertEqual(iou2, 0.0, accuracy: 0.001) // Non-overlapping boxes should have IOU = 0.0
        
        // Test partially overlapping rotated boxes
        let box4 = OBB(cx: 120, cy: 100, w: 50, h: 30, angle: Float.pi / 4)
        let iou3 = obbIoU(box1, box4)
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
        
        let intersection = polygonIntersection(subjectPolygon: square1, clipPolygon: square2)
        
        // The intersection should be a 5x5 square
        XCTAssertEqual(intersection.count, 4)
        let area = polygonArea(intersection)
        XCTAssertEqual(area, 25.0, accuracy: 0.001)
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndOBBDetection() {
        // Skip this test as it requires mocking VNCoreMLFeatureValueObservation
        XCTSkip("This test requires a real CoreML model and VNCoreMLFeatureValueObservation")
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