// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import CoreGraphics
@testable import YOLO

/// Comprehensive tests for NonMaxSuppression functionality
class NonMaxSuppressionTests: XCTestCase {
    
    // MARK: - Basic Functionality Tests
    
    func testNonMaxSuppressionWithNoBoxes() {
        // Test with empty arrays
        let boxes: [CGRect] = []
        let scores: [Float] = []
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        XCTAssertEqual(result.count, 0)
        XCTAssertEqual(result, [])
    }
    
    func testNonMaxSuppressionWithSingleBox() {
        // Test with single box - should always be selected
        let boxes = [CGRect(x: 100, y: 100, width: 50, height: 50)]
        let scores: [Float] = [0.9]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 0)
    }
    
    func testNonMaxSuppressionWithNonOverlappingBoxes() {
        // Test with boxes that don't overlap - all should be selected
        let boxes = [
            CGRect(x: 0, y: 0, width: 50, height: 50),
            CGRect(x: 100, y: 100, width: 50, height: 50),
            CGRect(x: 200, y: 200, width: 50, height: 50)
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(Set(result), Set([0, 1, 2]))
    }
    
    func testNonMaxSuppressionWithCompleteOverlap() {
        // Test with identical boxes - only highest score should be selected
        let box = CGRect(x: 100, y: 100, width: 50, height: 50)
        let boxes = [box, box, box]
        let scores: [Float] = [0.7, 0.9, 0.8]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 1) // Index 1 has highest score (0.9)
    }
    
    func testNonMaxSuppressionWithPartialOverlap() {
        // Test with partially overlapping boxes
        let boxes = [
            CGRect(x: 100, y: 100, width: 100, height: 100),
            CGRect(x: 150, y: 150, width: 100, height: 100), // 50% overlap with first
            CGRect(x: 300, y: 300, width: 100, height: 100)  // No overlap
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.3)
        
        // With threshold 0.3, boxes with >30% overlap should be suppressed
        // Box 1 overlaps with Box 0 by 25% of total area (2500/10000 = 0.25)
        // Since 0.25 < 0.3, both should be kept
        XCTAssertEqual(result.count, 3)
        
        // Test with lower threshold
        let resultLowThreshold = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.2)
        // Now box 1 should be suppressed
        XCTAssertEqual(resultLowThreshold.count, 2)
        XCTAssertTrue(resultLowThreshold.contains(0)) // Highest score
        XCTAssertTrue(resultLowThreshold.contains(2)) // No overlap
    }
    
    func testNonMaxSuppressionScoreOrdering() {
        // Test that boxes are selected in order of confidence score
        let boxes = [
            CGRect(x: 0, y: 0, width: 100, height: 100),
            CGRect(x: 200, y: 0, width: 100, height: 100),
            CGRect(x: 400, y: 0, width: 100, height: 100),
            CGRect(x: 600, y: 0, width: 100, height: 100)
        ]
        let scores: [Float] = [0.6, 0.9, 0.7, 0.8]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        // All boxes are non-overlapping, so all should be selected
        XCTAssertEqual(result.count, 4)
        
        // Verify they are in score order (indices: 1, 3, 2, 0)
        let sortedScores = result.map { scores[$0] }
        for i in 0..<sortedScores.count-1 {
            XCTAssertGreaterThanOrEqual(sortedScores[i], sortedScores[i+1])
        }
    }
    
    // MARK: - Threshold Tests
    
    func testNonMaxSuppressionWithZeroThreshold() {
        // With threshold 0, any overlap should cause suppression
        let boxes = [
            CGRect(x: 100, y: 100, width: 100, height: 100),
            CGRect(x: 199, y: 199, width: 100, height: 100) // 1 pixel overlap
        ]
        let scores: [Float] = [0.9, 0.8]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.0)
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 0) // Higher score wins
    }
    
    func testNonMaxSuppressionWithHighThreshold() {
        // With threshold 1.0, only complete overlap should cause suppression
        let boxes = [
            CGRect(x: 100, y: 100, width: 100, height: 100),
            CGRect(x: 100, y: 100, width: 100, height: 100), // Complete overlap
            CGRect(x: 150, y: 150, width: 100, height: 100)  // Partial overlap
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 1.0)
        
        // Only complete overlap should be suppressed
        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(result.contains(0)) // Highest score
        XCTAssertTrue(result.contains(2)) // Partial overlap not suppressed
    }
    
    // MARK: - Complex Scenarios
    
    func testNonMaxSuppressionChainEffect() {
        // Test that suppression doesn't cascade incorrectly
        let boxes = [
            CGRect(x: 0, y: 0, width: 100, height: 100),
            CGRect(x: 50, y: 0, width: 100, height: 100),   // Overlaps with 0
            CGRect(x: 100, y: 0, width: 100, height: 100),  // Overlaps with 1 but not 0
            CGRect(x: 150, y: 0, width: 100, height: 100)   // Overlaps with 2 but not 0 or 1
        ]
        let scores: [Float] = [0.9, 0.8, 0.7, 0.6]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.4)
        
        // Box 0 should suppress box 1 (50% overlap)
        // Box 2 doesn't overlap enough with box 0, so it stays
        // Box 3 doesn't overlap enough with boxes 0 or 2, so it stays
        XCTAssertEqual(result.count, 3)
        XCTAssertTrue(result.contains(0))
        XCTAssertTrue(result.contains(2))
        XCTAssertTrue(result.contains(3))
    }
    
    func testNonMaxSuppressionWithManyBoxes() {
        // Test performance with many boxes
        var boxes = [CGRect]()
        var scores = [Float]()
        
        // Create a grid of boxes
        for i in 0..<10 {
            for j in 0..<10 {
                boxes.append(CGRect(x: Double(i * 110), y: Double(j * 110), width: 100, height: 100))
                scores.append(Float.random(in: 0.1...0.9))
            }
        }
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        // All boxes are non-overlapping, so all should be selected
        XCTAssertEqual(result.count, 100)
    }
    
    func testNonMaxSuppressionWithDifferentSizedBoxes() {
        // Test with boxes of different sizes
        let boxes = [
            CGRect(x: 100, y: 100, width: 200, height: 200), // Large box
            CGRect(x: 150, y: 150, width: 50, height: 50),   // Small box inside large
            CGRect(x: 250, y: 250, width: 100, height: 100)  // Medium box partially overlapping
        ]
        let scores: [Float] = [0.8, 0.9, 0.7]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.3)
        
        // Small box has highest score and should be kept
        // Large box overlaps significantly with small box relative to small box's area
        XCTAssertTrue(result.contains(1)) // Small box with highest score
        
        // The algorithm uses min(area1, area2) for threshold calculation
        // Intersection between box 0 and 1 is 2500 (50x50)
        // Min area is 2500 (small box)
        // Ratio is 2500/2500 = 1.0 > 0.3, so box 0 should be suppressed
        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(result.contains(1))
        XCTAssertTrue(result.contains(2))
    }
    
    // MARK: - Edge Cases
    
    func testNonMaxSuppressionWithNegativeCoordinates() {
        // Test with boxes having negative coordinates
        let boxes = [
            CGRect(x: -50, y: -50, width: 100, height: 100),
            CGRect(x: -25, y: -25, width: 100, height: 100),
            CGRect(x: 100, y: 100, width: 100, height: 100)
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        // First two boxes overlap significantly
        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(result.contains(0)) // Highest score
        XCTAssertTrue(result.contains(2)) // No overlap
    }
    
    func testNonMaxSuppressionWithZeroAreaBoxes() {
        // Test with degenerate boxes (zero area)
        let boxes = [
            CGRect(x: 100, y: 100, width: 0, height: 100),   // Zero width
            CGRect(x: 200, y: 200, width: 100, height: 0),   // Zero height
            CGRect(x: 300, y: 300, width: 100, height: 100)  // Normal box
        ]
        let scores: [Float] = [0.9, 0.8, 0.7]
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        
        // Zero area boxes should still be processed
        XCTAssertEqual(result.count, 3)
    }
    
    func testNonMaxSuppressionWithEqualScores() {
        // Test behavior with equal scores
        let boxes = [
            CGRect(x: 100, y: 100, width: 100, height: 100),
            CGRect(x: 150, y: 150, width: 100, height: 100),
            CGRect(x: 300, y: 300, width: 100, height: 100)
        ]
        let scores: [Float] = [0.8, 0.8, 0.8] // All equal
        
        let result = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.3)
        
        // With equal scores, order depends on original indices
        // First box in list with highest score gets priority
        XCTAssertTrue(result.contains(0))
        XCTAssertTrue(result.contains(2)) // No overlap with 0
    }
    
    // MARK: - CGRect Extension Tests
    
    func testCGRectAreaExtension() {
        // Test the area calculation extension
        let rect1 = CGRect(x: 0, y: 0, width: 10, height: 20)
        XCTAssertEqual(rect1.area, 200)
        
        let rect2 = CGRect(x: 0, y: 0, width: 0, height: 20)
        XCTAssertEqual(rect2.area, 0)
        
        let rect3 = CGRect(x: 0, y: 0, width: -10, height: 20)
        XCTAssertEqual(rect3.area, -200) // Negative area for invalid rect
    }
    
    // MARK: - Performance Tests
    
    func testNonMaxSuppressionPerformance() {
        // Test performance with realistic scenario
        var boxes = [CGRect]()
        var scores = [Float]()
        
        // Create overlapping clusters of boxes
        for cluster in 0..<5 {
            let baseX = Double(cluster * 300)
            for i in 0..<20 {
                let x = baseX + Double(i * 5)
                let y = 100.0 + Double(i * 5)
                boxes.append(CGRect(x: x, y: y, width: 100, height: 100))
                scores.append(Float.random(in: 0.5...0.95))
            }
        }
        
        measure {
            let _ = nonMaxSuppression(boxes: boxes, scores: scores, threshold: 0.5)
        }
    }
}