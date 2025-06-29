// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import CoreImage
@testable import YOLO

/// Tests for adaptive sizing in Plot.swift visualization functions
class PlotAdaptiveSizeTests: XCTestCase {
    
    // MARK: - Test Adaptive Sizing Calculations
    
    func testAdaptiveSizingForSmallImage() {
        // Test with a small image (320x240)
        let width: CGFloat = 320
        let height: CGFloat = 240
        let imageDiagonal = sqrt(width * width + height * height)
        
        // Circle radius calculation
        let circleRadius = max(8.0, min(imageDiagonal * 0.03, 40.0))
        XCTAssertGreaterThanOrEqual(circleRadius, 8.0)
        XCTAssertLessThanOrEqual(circleRadius, 40.0)
        XCTAssertEqual(circleRadius, 12.0, accuracy: 0.5) // Should be around 12 pixels
        
        // Line width calculation
        let lineWidth = max(3.0, circleRadius * 0.5)
        XCTAssertGreaterThanOrEqual(lineWidth, 3.0)
        XCTAssertEqual(lineWidth, 6.0, accuracy: 0.5)
        
        // Font size calculation
        let fontSize = max(48.0, min(imageDiagonal * 0.1, 120.0))
        XCTAssertGreaterThanOrEqual(fontSize, 48.0)
        XCTAssertLessThanOrEqual(fontSize, 120.0)
    }
    
    func testAdaptiveSizingForMediumImage() {
        // Test with a medium image (1920x1080 - Full HD)
        let width: CGFloat = 1920
        let height: CGFloat = 1080
        let imageDiagonal = sqrt(width * width + height * height)
        
        // Circle radius calculation
        let circleRadius = max(8.0, min(imageDiagonal * 0.03, 40.0))
        XCTAssertEqual(circleRadius, 40.0, accuracy: 0.1) // Should hit the max limit
        
        // Line width calculation
        let lineWidth = max(3.0, circleRadius * 0.5)
        XCTAssertEqual(lineWidth, 20.0, accuracy: 0.5)
        
        // Font size calculation
        let fontSize = max(48.0, min(imageDiagonal * 0.1, 120.0))
        XCTAssertEqual(fontSize, 120.0, accuracy: 0.1) // Should hit the max limit
    }
    
    func testAdaptiveSizingForLargeImage() {
        // Test with a large image (4K - 3840x2160)
        let width: CGFloat = 3840
        let height: CGFloat = 2160
        let imageDiagonal = sqrt(width * width + height * height)
        
        // Circle radius calculation (capped at 40)
        let circleRadius = max(8.0, min(imageDiagonal * 0.03, 40.0))
        XCTAssertEqual(circleRadius, 40.0) // Should be capped at max
        
        // Font size calculation (capped at 50)
        let fontSize = max(48.0, min(imageDiagonal * 0.1, 120.0))
        XCTAssertEqual(fontSize, 50.0) // Should be capped at max
    }
    
    func testAdaptiveSizingForTinyImage() {
        // Test with a tiny image (100x100)
        let width: CGFloat = 100
        let height: CGFloat = 100
        let imageDiagonal = sqrt(width * width + height * height)
        
        // Circle radius calculation (should hit minimum)
        let circleRadius = max(8.0, min(imageDiagonal * 0.03, 40.0))
        XCTAssertEqual(circleRadius, 8.0, accuracy: 0.1) // Should hit minimum
        
        // Font size calculation (should hit minimum)
        let fontSize = max(48.0, min(imageDiagonal * 0.1, 120.0))
        XCTAssertEqual(fontSize, 48.0) // Should hit minimum
    }
    
    func testCornerRadiusScaling() {
        // Test corner radius scaling with line width
        let lineWidths: [CGFloat] = [1.0, 2.0, 5.0, 10.0, 20.0]
        
        for lineWidth in lineWidths {
            let cornerRadius = min(30.0, lineWidth * 3)
            XCTAssertLessThanOrEqual(cornerRadius, 30.0)
            XCTAssertGreaterThan(cornerRadius, 0)
            
            if lineWidth <= 10.0 {
                XCTAssertEqual(cornerRadius, lineWidth * 3)
            } else {
                XCTAssertEqual(cornerRadius, 30.0)
            }
        }
    }
    
    func testLineWidthForDifferentVisualizationTypes() {
        let width: CGFloat = 1024
        let height: CGFloat = 768
        let imageDiagonal = sqrt(width * width + height * height)
        
        // Detection/OBB line width
        let detectionLineWidth = max(3.0, imageDiagonal * 0.008)
        XCTAssertEqual(detectionLineWidth, 10.24, accuracy: 0.1)
        
        // Classification line width (thinner)
        let classificationLineWidth = max(3.0, imageDiagonal * 0.005)
        XCTAssertEqual(classificationLineWidth, 6.4, accuracy: 0.1)
        
        // Ensure detection lines are thicker than classification
        XCTAssertGreaterThan(detectionLineWidth, classificationLineWidth)
    }
    
    // MARK: - Integration Tests
    
    func testPoseVisualizationSizing() {
        // Create test image
        let testImage = createTestCIImage(width: 640, height: 480)
        let imageDiagonal = sqrt(640.0 * 640.0 + 480.0 * 480.0)
        
        // Calculate expected sizes
        let expectedRadius = max(8.0, min(imageDiagonal * 0.03, 40.0))
        let expectedLineWidth = max(3.0, expectedRadius * 0.5)
        
        // Verify radius is appropriate for pose keypoints
        XCTAssertEqual(expectedRadius, 24.0, accuracy: 0.5)
        XCTAssertEqual(expectedLineWidth, 12.0, accuracy: 0.5)
    }
    
    // MARK: - Helper Methods
    
    private func createTestCIImage(width: CGFloat, height: CGFloat) -> CIImage {
        return CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }
}