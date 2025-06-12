// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import UIKit
import CoreImage
@testable import YOLO

/// Minimal tests for YOLO main class functionality
class YOLOMainTests: XCTestCase {
    
    func testYOLOInitializationWithInvalidPath() {
        // Test YOLO initialization with invalid model path calls completion with error
        let expectation = XCTestExpectation(description: "Invalid model path")
        
        let _ = YOLO("invalid_model_path", task: .detect) { result in
            switch result {
            case .success(_):
                XCTFail("Should not succeed with invalid path")
            case .failure(let error):
                XCTAssertNotNil(error)
                if case PredictorError.modelFileNotFound = error {
                    // Expected error type
                } else {
                    XCTFail("Unexpected error type: \(error)")
                }
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    func testYOLOCallAsFunctionWithUIImage() {
        // Test YOLO callable interface with UIImage
        let yolo = createMockYOLO()
        
        // Create a test UIImage
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 100, height: 100), false, 1.0)
        UIColor.red.setFill()
        UIRectFill(CGRect(x: 0, y: 0, width: 100, height: 100))
        let testImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        let result = yolo(testImage)
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result.boxes.count, 0) // Mock predictor returns empty
        XCTAssertEqual(result.names.count, 0)
    }
    
    func testYOLOCallAsFunctionWithCIImage() {
        // Test YOLO callable interface with CIImage
        let yolo = createMockYOLO()
        
        let testImage = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 200, height: 200))
        let result = yolo(testImage)
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result.orig_shape, .zero) // Mock predictor returns zero size
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
    }
    
    func testYOLOCallAsFunctionWithCGImage() {
        // Test YOLO callable interface with CGImage
        let yolo = createMockYOLO()
        
        // Create a test CGImage
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 150, height: 100), false, 1.0)
        UIColor.green.setFill()
        UIRectFill(CGRect(x: 0, y: 0, width: 150, height: 100))
        let testUIImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        let testCGImage = testUIImage.cgImage!
        
        let result = yolo(testCGImage)
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result.boxes.count, 0)
    }
    
    func testYOLOCallAsFunctionWithResourceName() {
        // Test YOLO callable interface with resource name (will fail gracefully)
        let yolo = createMockYOLO()
        
        // This will fail because the resource doesn't exist, but should return empty result
        let result = yolo("nonexistent_image", withExtension: "jpg")
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result.orig_shape, .zero)
        XCTAssertEqual(result.boxes.count, 0)
        XCTAssertEqual(result.speed, 0, accuracy: 0.001)
        XCTAssertEqual(result.names.count, 0)
    }
    
    func testYOLOCallAsFunctionWithRemoteURL() {
        // Test YOLO callable interface with remote URL (will fail gracefully)
        let yolo = createMockYOLO()
        
        // This will fail because the URL doesn't exist, but should return empty result
        let invalidURL = URL(string: "https://nonexistent.example.com/image.jpg")
        let result = yolo(invalidURL)
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result.orig_shape, .zero)
        XCTAssertEqual(result.boxes.count, 0)
    }
    
    func testYOLOCallAsFunctionWithLocalPath() {
        // Test YOLO callable interface with local path (will fail gracefully)
        let yolo = createMockYOLO()
        
        // This will fail because the path doesn't exist, but should return empty result
        let result = yolo("/nonexistent/path/image.jpg")
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result.orig_shape, .zero)
        XCTAssertEqual(result.boxes.count, 0)
    }
    
    func testYOLOAllTaskTypes() {
        // Test YOLO initialization for all task types with invalid paths
        let tasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]
        
        for task in tasks {
            let expectation = XCTestExpectation(description: "Task \(task)")
            
            let _ = YOLO("invalid_path", task: task) { result in
                switch result {
                case .success(_):
                    XCTFail("Should not succeed with invalid path for task \(task)")
                case .failure(let error):
                    XCTAssertNotNil(error)
                    expectation.fulfill()
                }
            }
            
            wait(for: [expectation], timeout: 3.0)
        }
    }
    
    func testYOLOWithReturnAnnotatedImageFlag() {
        // Test YOLO callable interface with returnAnnotatedImage flag
        let yolo = createMockYOLO()
        
        let testImage = CIImage(color: .yellow).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        
        let resultWithAnnotation = yolo(testImage, returnAnnotatedImage: true)
        let resultWithoutAnnotation = yolo(testImage, returnAnnotatedImage: false)
        
        XCTAssertNotNil(resultWithAnnotation)
        XCTAssertNotNil(resultWithoutAnnotation)
        // Both should work regardless of flag with mock predictor
    }
    
    // MARK: - Helper Methods
    
    private func createMockYOLO() -> YOLO {
        // Create a mock YOLO instance for testing
        let yolo = YOLO.__allocating_init()
        yolo.predictor = MockPredictor()
        return yolo
    }
}

// MARK: - Mock Classes for Testing

class MockPredictor: Predictor {
    var labels: [String] = []
    var isUpdating: Bool = false
    
    func predict(sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?, onInferenceTime: InferenceTimeListener?) {
        // Mock implementation - do nothing
    }
    
    func predictOnImage(image: CIImage) -> YOLOResult {
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
}

// Extension to allow creating YOLO instances without going through the full init
extension YOLO {
    static func __allocating_init() -> YOLO {
        // This is a hack to create a YOLO instance for testing without model loading
        let yolo = unsafeBitCast(
            class_createInstance(YOLO.self, 0),
            to: YOLO.self
        )
        return yolo
    }
}
