// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import UIKit
@testable import YOLO

/// Comprehensive tests for BoundingBoxView functionality
@MainActor
class BoundingBoxViewTests: XCTestCase {
    
    var boundingBoxView: BoundingBoxView!
    
    override func setUp() {
        super.setUp()
        boundingBoxView = BoundingBoxView()
    }
    
    override func tearDown() {
        boundingBoxView = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testBoundingBoxViewInitialization() {
        // Verify initial state of shape layer
        XCTAssertNotNil(boundingBoxView.shapeLayer)
        XCTAssertEqual(boundingBoxView.shapeLayer.fillColor, UIColor.clear.cgColor)
        XCTAssertEqual(boundingBoxView.shapeLayer.lineWidth, 4.0)
        XCTAssertTrue(boundingBoxView.shapeLayer.isHidden)
        
        // Verify initial state of text layer
        XCTAssertNotNil(boundingBoxView.textLayer)
        XCTAssertTrue(boundingBoxView.textLayer.isHidden)
        XCTAssertEqual(boundingBoxView.textLayer.fontSize, 14)
        XCTAssertEqual(boundingBoxView.textLayer.alignmentMode, .center)
        XCTAssertEqual(boundingBoxView.textLayer.contentsScale, UIScreen.main.scale)
    }
    
    func testTextLayerFontConfiguration() {
        // Verify font configuration
        XCTAssertNotNil(boundingBoxView.textLayer.font)
        XCTAssertEqual(boundingBoxView.textLayer.fontSize, 14)
    }
    
    // MARK: - Line Width Tests
    
    func testSetLineWidth() {
        // Test setting valid line width
        boundingBoxView.setLineWidth(6.0)
        XCTAssertEqual(boundingBoxView.getLineWidth(), 6.0)
        XCTAssertEqual(boundingBoxView.shapeLayer.lineWidth, 6.0)
        
        // Test setting another valid width
        boundingBoxView.setLineWidth(2.5)
        XCTAssertEqual(boundingBoxView.getLineWidth(), 2.5)
        XCTAssertEqual(boundingBoxView.shapeLayer.lineWidth, 2.5)
    }
    
    func testSetLineWidthClamping() {
        // Test minimum clamping
        boundingBoxView.setLineWidth(0.5)
        XCTAssertEqual(boundingBoxView.getLineWidth(), 1.0) // Should be clamped to 1.0
        XCTAssertEqual(boundingBoxView.shapeLayer.lineWidth, 1.0)
        
        // Test maximum clamping
        boundingBoxView.setLineWidth(15.0)
        XCTAssertEqual(boundingBoxView.getLineWidth(), 10.0) // Should be clamped to 10.0
        XCTAssertEqual(boundingBoxView.shapeLayer.lineWidth, 10.0)
        
        // Test negative value
        boundingBoxView.setLineWidth(-5.0)
        XCTAssertEqual(boundingBoxView.getLineWidth(), 1.0) // Should be clamped to 1.0
    }
    
    func testGetLineWidth() {
        // Test getting default line width
        XCTAssertEqual(boundingBoxView.getLineWidth(), 4.0)
        
        // Test getting after setting
        boundingBoxView.setLineWidth(7.0)
        XCTAssertEqual(boundingBoxView.getLineWidth(), 7.0)
    }
    
    // MARK: - Layer Management Tests
    
    func testAddToLayer() {
        // Create a parent layer
        let parentLayer = CALayer()
        
        // Verify initial state
        XCTAssertEqual(parentLayer.sublayers?.count ?? 0, 0)
        
        // Add bounding box view layers
        boundingBoxView.addToLayer(parentLayer)
        
        // Verify layers were added
        XCTAssertEqual(parentLayer.sublayers?.count, 2)
        XCTAssertTrue(parentLayer.sublayers?.contains(boundingBoxView.shapeLayer) ?? false)
        XCTAssertTrue(parentLayer.sublayers?.contains(boundingBoxView.textLayer) ?? false)
    }
    
    func testAddToLayerOrder() {
        // Test that layers are added in correct order
        let parentLayer = CALayer()
        boundingBoxView.addToLayer(parentLayer)
        
        // Shape layer should be added before text layer
        let sublayers = parentLayer.sublayers ?? []
        if sublayers.count >= 2 {
            XCTAssertTrue(sublayers[0] === boundingBoxView.shapeLayer)
            XCTAssertTrue(sublayers[1] === boundingBoxView.textLayer)
        }
    }
    
    // MARK: - Show/Hide Tests
    
    func testShow() {
        // Test showing bounding box
        let frame = CGRect(x: 100, y: 100, width: 200, height: 150)
        let label = "person 95%"
        let color = UIColor.red
        let alpha: CGFloat = 0.8
        
        boundingBoxView.show(frame: frame, label: label, color: color, alpha: alpha)
        
        // Verify shape layer
        XCTAssertFalse(boundingBoxView.shapeLayer.isHidden)
        XCTAssertNotNil(boundingBoxView.shapeLayer.path)
        XCTAssertEqual(boundingBoxView.shapeLayer.strokeColor, color.withAlphaComponent(alpha).cgColor)
        
        // Verify text layer
        XCTAssertFalse(boundingBoxView.textLayer.isHidden)
        XCTAssertEqual(boundingBoxView.textLayer.string as? String, label)
        XCTAssertEqual(boundingBoxView.textLayer.backgroundColor, color.withAlphaComponent(alpha).cgColor)
        XCTAssertEqual(boundingBoxView.textLayer.foregroundColor, UIColor.white.withAlphaComponent(alpha).cgColor)
    }
    
    func testShowWithDifferentParameters() {
        // Test with different parameters
        let testCases: [(frame: CGRect, label: String, color: UIColor, alpha: CGFloat)] = [
            (CGRect(x: 0, y: 0, width: 50, height: 50), "cat 80%", .blue, 0.5),
            (CGRect(x: 200, y: 200, width: 300, height: 200), "car 99%", .green, 1.0),
            (CGRect(x: -50, y: -50, width: 100, height: 100), "dog 70%", .yellow, 0.3)
        ]
        
        for testCase in testCases {
            boundingBoxView.show(
                frame: testCase.frame,
                label: testCase.label,
                color: testCase.color,
                alpha: testCase.alpha
            )
            
            XCTAssertFalse(boundingBoxView.shapeLayer.isHidden)
            XCTAssertFalse(boundingBoxView.textLayer.isHidden)
            XCTAssertEqual(boundingBoxView.textLayer.string as? String, testCase.label)
        }
    }
    
    func testHide() {
        // First show the bounding box
        boundingBoxView.show(
            frame: CGRect(x: 100, y: 100, width: 100, height: 100),
            label: "test",
            color: .red,
            alpha: 0.8
        )
        
        // Verify it's visible
        XCTAssertFalse(boundingBoxView.shapeLayer.isHidden)
        XCTAssertFalse(boundingBoxView.textLayer.isHidden)
        
        // Hide it
        boundingBoxView.hide()
        
        // Verify it's hidden
        XCTAssertTrue(boundingBoxView.shapeLayer.isHidden)
        XCTAssertTrue(boundingBoxView.textLayer.isHidden)
    }
    
    // MARK: - Path and Frame Tests
    
    func testBoundingBoxPath() {
        // Test that the correct path is created
        let frame = CGRect(x: 50, y: 50, width: 100, height: 80)
        boundingBoxView.show(frame: frame, label: "test", color: .red, alpha: 1.0)
        
        // Verify path exists and has correct bounds
        XCTAssertNotNil(boundingBoxView.shapeLayer.path)
        
        let pathBounds = boundingBoxView.shapeLayer.path?.boundingBox ?? .zero
        XCTAssertEqual(pathBounds.origin.x, frame.origin.x, accuracy: 0.1)
        XCTAssertEqual(pathBounds.origin.y, frame.origin.y, accuracy: 0.1)
        XCTAssertEqual(pathBounds.width, frame.width, accuracy: 0.1)
        XCTAssertEqual(pathBounds.height, frame.height, accuracy: 0.1)
    }
    
    func testTextLayerPositioning() {
        // Test that text layer is positioned above bounding box
        let frame = CGRect(x: 100, y: 200, width: 150, height: 100)
        let label = "person 95%"
        
        boundingBoxView.show(frame: frame, label: label, color: .red, alpha: 1.0)
        
        // Text should be positioned above the bounding box
        XCTAssertLessThan(boundingBoxView.textLayer.frame.minY, frame.minY)
        XCTAssertEqual(boundingBoxView.textLayer.frame.minX, frame.minX - 2, accuracy: 0.1)
    }
    
    // MARK: - Text Rendering Tests
    
    func testLongLabelHandling() {
        // Test with a very long label
        let longLabel = "very_long_class_name_that_might_overflow 99.99%"
        let frame = CGRect(x: 100, y: 100, width: 100, height: 100)
        
        boundingBoxView.show(frame: frame, label: longLabel, color: .red, alpha: 1.0)
        
        XCTAssertEqual(boundingBoxView.textLayer.string as? String, longLabel)
        XCTAssertGreaterThan(boundingBoxView.textLayer.frame.width, 0)
    }
    
    func testEmptyLabelHandling() {
        // Test with empty label
        let frame = CGRect(x: 100, y: 100, width: 100, height: 100)
        
        boundingBoxView.show(frame: frame, label: "", color: .red, alpha: 1.0)
        
        XCTAssertEqual(boundingBoxView.textLayer.string as? String, "")
        XCTAssertFalse(boundingBoxView.textLayer.isHidden)
    }
    
    // MARK: - Color and Alpha Tests
    
    func testColorAndAlphaApplication() {
        // Test various color and alpha combinations
        let frame = CGRect(x: 0, y: 0, width: 100, height: 100)
        let testCases: [(color: UIColor, alpha: CGFloat)] = [
            (.red, 1.0),
            (.blue, 0.5),
            (.green, 0.0),
            (.black, 0.7)
        ]
        
        for testCase in testCases {
            boundingBoxView.show(frame: frame, label: "test", color: testCase.color, alpha: testCase.alpha)
            
            let expectedColor = testCase.color.withAlphaComponent(testCase.alpha).cgColor
            XCTAssertEqual(boundingBoxView.shapeLayer.strokeColor, expectedColor)
            XCTAssertEqual(boundingBoxView.textLayer.backgroundColor, expectedColor)
            
            let expectedTextColor = UIColor.white.withAlphaComponent(testCase.alpha).cgColor
            XCTAssertEqual(boundingBoxView.textLayer.foregroundColor, expectedTextColor)
        }
    }
    
    // MARK: - Transaction Tests
    
    func testCATransactionDisabling() {
        // Test that animations are disabled during show
        // This is harder to test directly, but we can verify the layers update immediately
        let frame = CGRect(x: 100, y: 100, width: 100, height: 100)
        
        boundingBoxView.show(frame: frame, label: "test", color: .red, alpha: 1.0)
        
        // Verify immediate update without animation
        XCTAssertFalse(boundingBoxView.shapeLayer.isHidden)
        XCTAssertFalse(boundingBoxView.textLayer.isHidden)
    }
    
    // MARK: - Edge Cases
    
    func testShowWithZeroSizeFrame() {
        // Test with zero-size frame
        let frame = CGRect(x: 100, y: 100, width: 0, height: 0)
        
        boundingBoxView.show(frame: frame, label: "test", color: .red, alpha: 1.0)
        
        XCTAssertFalse(boundingBoxView.shapeLayer.isHidden)
        XCTAssertNotNil(boundingBoxView.shapeLayer.path)
    }
    
    func testShowWithNegativeSizeFrame() {
        // Test with negative size frame
        let frame = CGRect(x: 100, y: 100, width: -50, height: -50)
        
        boundingBoxView.show(frame: frame, label: "test", color: .red, alpha: 1.0)
        
        // Should still create a path, even if it's invalid
        XCTAssertNotNil(boundingBoxView.shapeLayer.path)
    }
    
    func testRapidShowHideCycles() {
        // Test rapid show/hide cycles
        let frame = CGRect(x: 100, y: 100, width: 100, height: 100)
        
        for i in 0..<10 {
            if i % 2 == 0 {
                boundingBoxView.show(frame: frame, label: "test \(i)", color: .red, alpha: 1.0)
                XCTAssertFalse(boundingBoxView.shapeLayer.isHidden)
            } else {
                boundingBoxView.hide()
                XCTAssertTrue(boundingBoxView.shapeLayer.isHidden)
            }
        }
    }
    
    // MARK: - Memory Management Tests
    
    func testLayerReuse() {
        // Test that layers are reused, not recreated
        let shapeLayerBefore = boundingBoxView.shapeLayer
        let textLayerBefore = boundingBoxView.textLayer
        
        // Show and hide multiple times
        for _ in 0..<5 {
            boundingBoxView.show(
                frame: CGRect(x: 0, y: 0, width: 100, height: 100),
                label: "test",
                color: .red,
                alpha: 1.0
            )
            boundingBoxView.hide()
        }
        
        // Verify same layer instances are used
        XCTAssertTrue(boundingBoxView.shapeLayer === shapeLayerBefore)
        XCTAssertTrue(boundingBoxView.textLayer === textLayerBefore)
    }
}