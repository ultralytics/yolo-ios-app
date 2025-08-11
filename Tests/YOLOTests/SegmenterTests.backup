// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

/// Minimal tests for Segmenter functionality
class SegmenterTests: XCTestCase {
    
    func testSegmenterInitialization() {
        // Test Segmenter initialization inherits from BasePredictor
        let segmenter = Segmenter()
        
        XCTAssertFalse(segmenter.isModelLoaded)
        XCTAssertEqual(segmenter.labels.count, 0)
        XCTAssertEqual(segmenter.confidenceThreshold, 0.25, accuracy: 0.001)
        XCTAssertEqual(segmenter.iouThreshold, 0.4, accuracy: 0.001)
        XCTAssertEqual(segmenter.numItemsThreshold, 30)
        XCTAssertFalse(segmenter.isUpdating)
    }
    
    func testSegmenterPredictOnImageWithoutModel() {
        // Test predictOnImage without loaded model returns empty result
        let segmenter = Segmenter()
        segmenter.labels = ["person", "car", "dog"]
        
        let image = CIImage(color: CIColor(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0)).cropped(to: CGRect(x: 0, y: 0, width: 640, height: 640))
        let result = segmenter.predictOnImage(image: image)
        
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertNil(result.probs)
        XCTAssertNil(result.masks) // Will be nil without model
        XCTAssertEqual(result.keypointsList.count, 0)
        XCTAssertEqual(result.names, ["person", "car", "dog"])
        XCTAssertEqual(result.orig_shape.width, 640)
        XCTAssertEqual(result.orig_shape.height, 640)
    }
    
    func testSegmenterProcessObservationsWithoutModel() {
        // Test processObservations without crashing
        let segmenter = Segmenter()
        segmenter.labels = ["person", "car", "bicycle"]
        segmenter.inputSize = CGSize(width: 640, height: 480)
        
        let mockRequest = MockVNRequest()
        
        // Should not crash
        segmenter.processObservations(for: mockRequest, error: nil)
        segmenter.processObservations(for: mockRequest, error: NSError(domain: "test", code: 1))
    }
    
    func testSegmenterLabelsAssignment() {
        // Test labels can be assigned and retrieved
        let segmenter = Segmenter()
        let testLabels = ["person", "bicycle", "car", "motorbike", "bus"]
        
        segmenter.labels = testLabels
        XCTAssertEqual(segmenter.labels, testLabels)
        XCTAssertEqual(segmenter.labels.count, 5)
    }
    
    func testSegmenterInputSize() {
        // Test input size can be set and retrieved
        let segmenter = Segmenter()
        let testSize = CGSize(width: 640, height: 640)
        
        segmenter.inputSize = testSize
        XCTAssertEqual(segmenter.inputSize, testSize)
    }
    
    func testSegmenterTimingProperties() {
        // Test timing properties are properly initialized
        let segmenter = Segmenter()
        
        XCTAssertEqual(segmenter.t0, 0.0, accuracy: 0.001)
        XCTAssertEqual(segmenter.t1, 0.0, accuracy: 0.001)
        XCTAssertEqual(segmenter.t2, 0.0, accuracy: 0.001)
        XCTAssertEqual(segmenter.t4, 0.0, accuracy: 0.001)
        XCTAssertGreaterThan(segmenter.t3, 0)
    }
    
    func testSegmenterIsInstanceOfBasePredictor() {
        // Test Segmenter is instance of BasePredictor
        let segmenter = Segmenter()
        
        XCTAssertNotNil(segmenter, "Segmenter should not be nil")
        XCTAssertTrue(type(of: segmenter) == Segmenter.self, "Should be Segmenter type")
    }
    
    func testSegmenterResultStructure() {
        // Test Segmenter result has correct structure
        let segmenter = Segmenter()
        segmenter.labels = ["person", "car"]
        
        let image = CIImage(color: CIColor(red: 0.0, green: 1.0, blue: 1.0, alpha: 1.0)).cropped(to: CGRect(x: 0, y: 0, width: 416, height: 416))
        let result = segmenter.predictOnImage(image: image)
        
        XCTAssertNotNil(result.boxes)
        XCTAssertNil(result.probs) // Segmentation doesn't use probs
        // result.masks could be nil without model
        XCTAssertEqual(result.keypointsList.count, 0) // Segmentation doesn't use keypoints
        XCTAssertEqual(result.obb.count, 0) // Segmentation doesn't use OBB
        XCTAssertEqual(result.names, ["person", "car"])
    }
    
    func testSegmenterColorsForMaskProperty() {
        // Test colorsForMask property exists and can be modified
        let segmenter = Segmenter()
        
        XCTAssertEqual(segmenter.colorsForMask.count, 0)
        
        segmenter.colorsForMask = [(255, 128, 0), (128, 255, 0), (0, 128, 255)]
        XCTAssertEqual(segmenter.colorsForMask.count, 3)
        XCTAssertEqual(segmenter.colorsForMask[0].red, 255)
        XCTAssertEqual(segmenter.colorsForMask[1].green, 255)
        XCTAssertEqual(segmenter.colorsForMask[2].blue, 255)
    }
    
    func testSegmenterModelInputSize() {
        // Test model input size properties
        let segmenter = Segmenter()
        
        XCTAssertEqual(segmenter.modelInputSize.width, 0)
        XCTAssertEqual(segmenter.modelInputSize.height, 0)
    }
    
    func testSegmenterCheckShapeDimensions() {
        // Test checkShapeDimensions utility method
        let segmenter = Segmenter()
        
        // Create a test MLMultiArray
        let testArray = try! MLMultiArray(shape: [1, 32, 160, 160], dataType: .float32)
        let dimensions = segmenter.checkShapeDimensions(of: testArray)
        
        XCTAssertEqual(dimensions, 4)
        
        let testArray2D = try! MLMultiArray(shape: [10, 5], dataType: .float32)
        let dimensions2D = segmenter.checkShapeDimensions(of: testArray2D)
        
        XCTAssertEqual(dimensions2D, 2)
    }
    
    func testSegmenterAdjustBox() {
        // Test adjustBox utility method
        let segmenter = Segmenter()
        let originalBox = CGRect(x: 100, y: 100, width: 200, height: 150)
        let containerSize = CGSize(width: 1280, height: 960)
        
        let adjustedBox = segmenter.adjustBox(originalBox, toFitIn: containerSize)
        
        XCTAssertEqual(adjustedBox.origin.x, 200, accuracy: 0.1) // 100 * (1280/640)
        XCTAssertEqual(adjustedBox.origin.y, 150, accuracy: 0.1) // 100 * (960/640)
        XCTAssertEqual(adjustedBox.width, 400, accuracy: 0.1) // 200 * (1280/640)
        XCTAssertEqual(adjustedBox.height, 225, accuracy: 0.1) // 150 * (960/640)
    }
}
