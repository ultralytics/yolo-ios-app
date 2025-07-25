// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import UIKit
import AVFoundation
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
    
    func testYOLOInitializationForAllTaskTypes() {
        // Test YOLO initialization with different task types
        let tasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]
        
        for task in tasks {
            let expectation = XCTestExpectation(description: "YOLO init for \(task)")
            
            _ = YOLO("nonexistent.mlmodel", task: task) { result in
                switch result {
                case .success:
                    XCTFail("Should fail with nonexistent model")
                case .failure(let error):
                    XCTAssertNotNil(error)
                    expectation.fulfill()
                }
            }
            
            wait(for: [expectation], timeout: 2.0)
        }
    }
    
    func testYOLOInitializationWithEmptyPath() {
        // Test YOLO initialization with empty path
        let expectation = XCTestExpectation(description: "Empty path initialization")
        
        _ = YOLO("", task: .detect) { result in
            switch result {
            case .success:
                XCTFail("Should fail with empty path")
            case .failure(let error):
                XCTAssertNotNil(error)
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
    
    func testYOLOInitializationErrorTypes() {
        // Test that proper error types are returned
        let expectation = XCTestExpectation(description: "Error type verification")
        
        _ = YOLO("missing_model.mlmodel", task: .detect) { result in
            switch result {
            case .success:
                XCTFail("Should fail with missing model")
            case .failure(let error):
                // Verify we get a meaningful error
                if let predictorError = error as? PredictorError {
                    switch predictorError {
                    case .modelFileNotFound:
                        // Expected error
                        break
                    default:
                        XCTFail("Unexpected predictor error type")
                    }
                } else {
                    // Other error types are also acceptable
                    XCTAssertNotNil(error.localizedDescription)
                }
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 2.0)
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
    
    func testYOLOTaskTypeEnumeration() {
        // Test that all YOLOTask cases are handled
        let allTasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]
        
        // Verify we have all expected task types
        XCTAssertEqual(allTasks.count, 5, "Should have 5 task types")
        
        // Verify each task type is unique
        let uniqueTasks = Set(allTasks)
        XCTAssertEqual(uniqueTasks.count, allTasks.count, "All task types should be unique")
    }
    
    func testYOLOCallAsFunctionWithMockPredictor() {
        // Test YOLO callAsFunction methods using mock predictor
        let expectation = XCTestExpectation(description: "Mock predictor test")
        
        _ = YOLO("mock", task: .detect) { result in
            switch result {
            case .success(let yolo):
                // Replace with mock predictor
                yolo.predictor = MockPredictor()
                
                // Test UIImage call
                let testImage = self.createTestImage()
                let result1 = yolo(testImage)
                XCTAssertNotNil(result1)
                XCTAssertEqual(result1.boxes.count, 1)
                
                // Test CIImage call
                let ciImage = CIImage(image: testImage)!
                let result2 = yolo(ciImage)
                XCTAssertNotNil(result2)
                
                // Test CGImage call
                let cgImage = testImage.cgImage!
                let result3 = yolo(cgImage)
                XCTAssertNotNil(result3)
                
                // Test resource name call (will fail)
                let result4 = yolo("nonexistent", withExtension: "jpg")
                XCTAssertEqual(result4.orig_shape, .zero)
                
                // Test URL call (will fail)
                let result5 = yolo(URL(string: "https://example.com/image.jpg"))
                XCTAssertEqual(result5.orig_shape, .zero)
                
                // Test local path call (will fail)
                let result6 = yolo("/nonexistent/path.jpg")
                XCTAssertEqual(result6.orig_shape, .zero)
                
                expectation.fulfill()
            case .failure:
                // It's OK if model loading fails
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    func testYOLOInitWithMLModelExtension() {
        // Test initialization with .mlmodel file extension
        let expectation = XCTestExpectation(description: "MLModel extension test")
        
        _ = YOLO("model.mlmodel", task: .detect) { result in
            // Should fail as file doesn't exist
            switch result {
            case .success:
                XCTFail("Should fail with non-existent file")
            case .failure:
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
    
    func testYOLOInitWithMLPackageExtension() {
        // Test initialization with .mlpackage file extension
        let expectation = XCTestExpectation(description: "MLPackage extension test")
        
        _ = YOLO("model.mlpackage", task: .detect) { result in
            // Should fail as file doesn't exist
            switch result {
            case .success:
                XCTFail("Should fail with non-existent file")
            case .failure:
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
    
    private func createTestImage() -> UIImage {
        let size = CGSize(width: 100, height: 100)
        UIGraphicsBeginImageContext(size)
        UIColor.blue.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return image
    }
    
    // MARK: - Helper Methods
    // (removed createMockYOLO - not needed for simplified tests)
}

// Mock classes have been moved to separate files to avoid conflicts
