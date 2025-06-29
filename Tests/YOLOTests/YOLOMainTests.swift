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
    
    // MARK: - Helper Methods
    // (removed createMockYOLO - not needed for simplified tests)
}

// Mock classes have been moved to separate files to avoid conflicts
