// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Comprehensive tests for Segmenter functionality
class SegmenterTests: XCTestCase {
    
    var segmenter: Segmenter!
    
    override func setUp() {
        super.setUp()
        segmenter = Segmenter()
    }
    
    override func tearDown() {
        segmenter = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testSegmenterInitialization() {
        // Verify Segmenter inherits proper initialization from BasePredictor
        XCTAssertFalse(segmenter.isModelLoaded)
        XCTAssertEqual(segmenter.labels.count, 0)
        XCTAssertEqual(segmenter.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(segmenter.iouThreshold, 0.4, accuracy: 0.001)
        XCTAssertEqual(segmenter.colorsForMask.count, 0)
    }
    
    // MARK: - Process Observations Tests
    
    func testProcessObservationsWithEmptyResults() {
        // Test processing with no segmentation results
        let request = MockVNRequestWithResults(results: [])
        
        // Should not crash
        segmenter.processObservations(for: request, error: nil)
    }
    
    func testProcessObservationsWithWrongNumberOfOutputs() {
        // Skip this test as it requires mocking VNCoreMLFeatureValueObservation
        XCTSkip("This test requires a real CoreML model and VNCoreMLFeatureValueObservation")
    }
    
    func testProcessObservationsWithValidSegmentationResults() {
        // Skip this test as it requires mocking VNCoreMLFeatureValueObservation
        XCTSkip("This test requires a real CoreML model and VNCoreMLFeatureValueObservation")
    }
    
    // MARK: - Shape Dimension Tests
    
    func testCheckShapeDimensions() {
        // Test shape dimension checking
        let shape3D = [1, 32, 100] as [NSNumber]
        let array3D = createMockMLMultiArray(shape: shape3D, values: [0.0])
        XCTAssertEqual(segmenter.checkShapeDimensions(of: array3D), 3)
        
        let shape4D = [1, 32, 160, 160] as [NSNumber]
        let array4D = createMockMLMultiArray(shape: shape4D, values: [0.0])
        XCTAssertEqual(segmenter.checkShapeDimensions(of: array4D), 4)
    }
    
    // MARK: - Post Process Segment Tests
    
    func testPostProcessSegmentWithNoDetections() {
        // Test post-processing with low confidence values (no detections)
        let numAnchors = 10
        let totalFeatures = 4 + 3 + 32 // box + classes + mask
        let shape = [1, totalFeatures, numAnchors] as [NSNumber]
        
        // Create values with very low confidence
        var values = [Double]()
        for _ in 0..<(totalFeatures * numAnchors) {
            values.append(Double.random(in: 0.0...0.1)) // Low confidence values
        }
        
        let multiArray = createMockMLMultiArray(shape: shape, values: values)
        
        let results = segmenter.postProcessSegment(
            feature: multiArray,
            confidenceThreshold: 0.5,
            iouThreshold: 0.4
        )
        
        XCTAssertEqual(results.count, 0) // No detections should pass threshold
    }
    
    func testPostProcessSegmentWithMultipleDetections() {
        // Test post-processing with multiple high-confidence detections
        let numAnchors = 5
        let numClasses = 3
        let totalFeatures = 4 + numClasses + 32
        let shape = [1, totalFeatures, numAnchors] as [NSNumber]
        
        let values = createHighConfidencePredictionValues(
            numAnchors: numAnchors,
            numClasses: numClasses
        )
        
        let multiArray = createMockMLMultiArray(shape: shape, values: values)
        
        let results = segmenter.postProcessSegment(
            feature: multiArray,
            confidenceThreshold: 0.3,
            iouThreshold: 0.4
        )
        
        XCTAssertGreaterThan(results.count, 0)
        
        // Verify result structure
        for result in results {
            let (box, classIndex, confidence, maskProbs) = result
            XCTAssertGreaterThan(box.width, 0)
            XCTAssertGreaterThan(box.height, 0)
            XCTAssertGreaterThanOrEqual(classIndex, 0)
            XCTAssertLessThan(classIndex, numClasses)
            XCTAssertGreaterThan(confidence, 0.3)
            XCTAssertEqual(maskProbs.count, 32)
        }
    }
    
    // MARK: - Coordinate Adjustment Tests
    
    func testAdjustBox() {
        // Test bounding box coordinate adjustment
        let originalBox = CGRect(x: 100, y: 100, width: 200, height: 200)
        let containerSize = CGSize(width: 1280, height: 960)
        
        let adjustedBox = segmenter.adjustBox(originalBox, toFitIn: containerSize)
        
        // Expected scaling: 1280/640 = 2.0 for x, 960/640 = 1.5 for y
        XCTAssertEqual(adjustedBox.origin.x, 200, accuracy: 0.01)
        XCTAssertEqual(adjustedBox.origin.y, 150, accuracy: 0.01)
        XCTAssertEqual(adjustedBox.width, 400, accuracy: 0.01)
        XCTAssertEqual(adjustedBox.height, 300, accuracy: 0.01)
    }
    
    // MARK: - Predict on Image Tests
    
    func testPredictOnImageWithNoModel() {
        // Test prediction when no model is loaded
        let image = createTestImage()
        
        let result = segmenter.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertNil(result.masks)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
    }
    
    func testPredictOnImageSetsInputSize() {
        // Test that predictOnImage properly sets input size
        let image = createTestImage(width: 1024, height: 768)
        
        _ = segmenter.predictOnImage(image: image)
        
        XCTAssertEqual(segmenter.inputSize.width, 1024)
        XCTAssertEqual(segmenter.inputSize.height, 768)
    }
    
    // MARK: - Performance Metrics Tests
    
    func testUpdateTime() {
        // Skip this test as it requires mocking VNCoreMLFeatureValueObservation with valid multiArrayValue
        XCTSkip("This test requires a real CoreML model and VNCoreMLFeatureValueObservation with valid predictions")
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndSegmentation() {
        // Test complete segmentation flow with mock data
        segmenter.labels = ["person", "bicycle", "car"]
        segmenter.setConfidenceThreshold(confidence: 0.4)
        segmenter.setIouThreshold(iou: 0.5)
        segmenter.modelInputSize = (width: 640, height: 640)
        
        let numAnchors = 20
        let numClasses = 3
        let totalFeatures = 4 + numClasses + 32
        
        // Create realistic prediction data
        let predShape = [1, totalFeatures, numAnchors] as [NSNumber]
        let predValues = createRealisticPredictionValues(
            numAnchors: numAnchors,
            numClasses: numClasses
        )
        let predArray = createMockMLMultiArray(shape: predShape, values: predValues)
        
        // Create mask prototypes
        let maskShape = [1, 32, 160, 160] as [NSNumber]
        let maskValues = createRealisticMaskValues()
        let maskArray = createMockMLMultiArray(shape: maskShape, values: maskValues)
        
        // Test shape detection
        let pred4D = segmenter.checkShapeDimensions(of: maskArray)
        let pred3D = segmenter.checkShapeDimensions(of: predArray)
        
        XCTAssertEqual(pred4D, 4)
        XCTAssertEqual(pred3D, 3)
        
        // Test post-processing
        let results = segmenter.postProcessSegment(
            feature: predArray,
            confidenceThreshold: 0.4,
            iouThreshold: 0.5
        )
        
        XCTAssertGreaterThan(results.count, 0)
        
        // Verify each detection
        for (box, classIdx, conf, maskProbs) in results {
            XCTAssertGreaterThan(box.width, 0)
            XCTAssertGreaterThan(box.height, 0)
            XCTAssertGreaterThanOrEqual(conf, 0.4)
            XCTAssertLessThan(classIdx, numClasses)
            XCTAssertEqual(maskProbs.count, 32)
        }
    }
    
    // MARK: - Helper Methods
    
    private func createTestImage(width: CGFloat = 640, height: CGFloat = 480) -> CIImage {
        return CIImage(color: CIColor(red: 1.0, green: 0.0, blue: 0.0)).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
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
    
    private func createMockPredictionValues(numAnchors: Int, numClasses: Int) -> [Double] {
        var values = [Double]()
        let totalFeatures = 4 + numClasses + 32
        
        for anchor in 0..<numAnchors {
            // Box coordinates (x, y, w, h)
            values.append(Double(320 + anchor * 10)) // x
            values.append(Double(240 + anchor * 10)) // y
            values.append(Double(50 + anchor * 5))  // width
            values.append(Double(50 + anchor * 5))  // height
            
            // Class probabilities
            for cls in 0..<numClasses {
                if anchor % numClasses == cls {
                    values.append(0.8) // High confidence for one class
                } else {
                    values.append(0.1)
                }
            }
            
            // Mask coefficients
            for _ in 0..<32 {
                values.append(Double.random(in: -1.0...1.0))
            }
        }
        
        return values
    }
    
    private func createHighConfidencePredictionValues(numAnchors: Int, numClasses: Int) -> [Double] {
        var values = [Double]()
        
        for anchor in 0..<numAnchors {
            // Box coordinates
            values.append(Double(100 + anchor * 100)) // x
            values.append(Double(100 + anchor * 80))  // y
            values.append(Double(80))  // width
            values.append(Double(80))  // height
            
            // Class probabilities with one high confidence
            for cls in 0..<numClasses {
                if anchor == cls {
                    values.append(0.9) // High confidence
                } else {
                    values.append(0.05)
                }
            }
            
            // Mask coefficients
            for _ in 0..<32 {
                values.append(Double.random(in: -0.5...0.5))
            }
        }
        
        return values
    }
    
    private func createRealisticPredictionValues(numAnchors: Int, numClasses: Int) -> [Double] {
        var values = [Double]()
        
        for anchor in 0..<numAnchors {
            // Realistic box coordinates
            let x = Double.random(in: 50...590)
            let y = Double.random(in: 50...590)
            let w = Double.random(in: 20...200)
            let h = Double.random(in: 20...200)
            
            values.append(x)
            values.append(y)
            values.append(w)
            values.append(h)
            
            // Realistic class distribution
            var classProbs = [Double](repeating: 0.01, count: numClasses)
            if anchor < 5 { // Only first 5 anchors have high confidence
                let mainClass = anchor % numClasses
                classProbs[mainClass] = Double.random(in: 0.7...0.95)
            }
            values.append(contentsOf: classProbs)
            
            // Mask coefficients
            for _ in 0..<32 {
                values.append(Double.random(in: -2.0...2.0))
            }
        }
        
        return values
    }
    
    private func createRealisticMaskValues() -> [Double] {
        var values = [Double]()
        
        // Create 32 prototype masks of size 160x160
        for _ in 0..<32 {
            for _ in 0..<(160 * 160) {
                values.append(Double.random(in: -1.0...1.0))
            }
        }
        
        return values
    }
}

// MARK: - FloatPointerWrapper Tests

extension SegmenterTests {
    func testFloatPointerWrapper() {
        // Test FloatPointerWrapper functionality
        var floatArray: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        floatArray.withUnsafeMutableBufferPointer { buffer in
            guard let pointer = buffer.baseAddress else {
                XCTFail("Failed to get pointer")
                return
            }
            
            let wrapper = FloatPointerWrapper(pointer)
            
            XCTAssertEqual(wrapper.pointer[0], 1.0)
            XCTAssertEqual(wrapper.pointer[1], 2.0)
            XCTAssertEqual(wrapper.pointer[4], 5.0)
        }
    }
    
    func testSegmenterAdditionalThresholds() {
        // Test additional threshold scenarios
        segmenter.setConfidenceThreshold(confidence: 0.99)
        segmenter.setIoUThreshold(iou: 0.01)
        
        // Test with extreme precision values
        segmenter.setConfidenceThreshold(confidence: 0.123456789)
        segmenter.setIoUThreshold(iou: 0.987654321)
        
        XCTAssertNotNil(segmenter)
    }
    
    func testPredictOnImageEdgeCases() {
        // Test edge cases for predictOnImage
        segmenter.labels = ["test"]
        
        // Very small image
        let tinyImage = CIImage(color: CIColor.green).cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))
        let result1 = segmenter.predictOnImage(image: tinyImage)
        XCTAssertEqual(result1.orig_shape, CGSize(width: 1, height: 1))
        
        // Non-square image
        let wideImage = CIImage(color: CIColor.yellow).cropped(to: CGRect(x: 0, y: 0, width: 1920, height: 480))
        let result2 = segmenter.predictOnImage(image: wideImage)
        XCTAssertEqual(result2.orig_shape, CGSize(width: 1920, height: 480))
    }
}