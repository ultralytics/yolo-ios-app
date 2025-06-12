// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import CoreGraphics
@testable import YOLO

/// Minimal tests for NonMaxSuppression algorithm
class NonMaxSuppressionTests: XCTestCase {
    
    func testCGRectAreaExtension() {
        // Test CGRect area calculation extension
        let rect = CGRect(x: 0, y: 0, width: 10, height: 5)
        XCTAssertEqual(rect.area, 50, accuracy: 0.001)
        
        let zeroRect = CGRect.zero
        XCTAssertEqual(zeroRect.area, 0, accuracy: 0.001)
    }
    
    func testNMSWithNoOverlap() {
        // Test NMS with non-overlapping boxes
        let boxes = [
            CGRect(x: 0, y: 0, width: 10, height: 10),
            CGRect(x: 20, y: 20, width: 10, height: 10),
            CGRect(x: 40, y: 40, width: 10, height: 10)
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let selected = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        XCTAssertEqual(selected.count, 3) // All boxes should be kept
        XCTAssertTrue(selected.contains(0))
        XCTAssertTrue(selected.contains(1))
        XCTAssertTrue(selected.contains(2))
    }
    
    func testNMSWithOverlap() {
        // Test NMS with overlapping boxes
        let boxes = [
            CGRect(x: 0, y: 0, width: 10, height: 10),
            CGRect(x: 5, y: 5, width: 10, height: 10), // Overlapping
            CGRect(x: 20, y: 20, width: 10, height: 10)
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let selected = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.3)
        
        XCTAssertEqual(selected.count, 2) // One box should be suppressed
        XCTAssertTrue(selected.contains(0)) // Highest score should be kept
        XCTAssertTrue(selected.contains(2)) // Non-overlapping should be kept
        XCTAssertFalse(selected.contains(1)) // Overlapping lower score should be removed
    }
    
    func testNMSWithIdenticalBoxes() {
        // Test NMS with identical boxes
        let boxes = [
            CGRect(x: 0, y: 0, width: 10, height: 10),
            CGRect(x: 0, y: 0, width: 10, height: 10)
        ]
        let scores: [Float] = [0.9, 0.8]
        
        let selected = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        XCTAssertEqual(selected.count, 1) // Only highest score should remain
        XCTAssertEqual(selected[0], 0)
    }
    
    func testNMSWithSingleBox() {
        // Test NMS with single box
        let boxes = [CGRect(x: 0, y: 0, width: 10, height: 10)]
        let scores: [Float] = [0.9]
        
        let selected = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        XCTAssertEqual(selected.count, 1)
        XCTAssertEqual(selected[0], 0)
    }
    
    func testNMSWithEmptyInput() {
        // Test NMS with empty input
        let boxes: [CGRect] = []
        let scores: [Float] = []
        
        let selected = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        XCTAssertEqual(selected.count, 0)
    }
    
    func testNMSThresholdSensitivity() {
        // Test NMS threshold sensitivity
        let boxes = [
            CGRect(x: 0, y: 0, width: 10, height: 10),
            CGRect(x: 5, y: 5, width: 10, height: 10)
        ]
        let scores: [Float] = [0.9, 0.8]
        
        // Low threshold - should suppress more
        let lowThreshold = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.1)
        XCTAssertEqual(lowThreshold.count, 1)
        
        // High threshold - should suppress less
        let highThreshold = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.9)
        XCTAssertEqual(highThreshold.count, 2)
    }
}
