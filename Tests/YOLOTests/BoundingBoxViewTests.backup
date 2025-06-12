// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import UIKit
import QuartzCore
@testable import YOLO

/// Minimal tests for BoundingBoxView functionality
@MainActor
class BoundingBoxViewTests: XCTestCase {

    func testBoundingBoxViewInitialization() {
        // Test BoundingBoxView initialization
        let boxView = BoundingBoxView()

        XCTAssertNotNil(boxView.shapeLayer)
        XCTAssertNotNil(boxView.textLayer)

        // Test shape layer properties
        XCTAssertEqual(boxView.shapeLayer.fillColor, UIColor.clear.cgColor)
        XCTAssertEqual(boxView.shapeLayer.lineWidth, 4)
        XCTAssertTrue(boxView.shapeLayer.isHidden)

        // Test text layer properties
        XCTAssertTrue(boxView.textLayer.isHidden)
        XCTAssertEqual(boxView.textLayer.contentsScale, UIScreen.main.scale)
        XCTAssertEqual(boxView.textLayer.fontSize, 14)
        XCTAssertEqual(boxView.textLayer.alignmentMode, .center)
    }

    func testBoundingBoxViewAddToLayer() {
        // Test BoundingBoxView addToLayer method
        let boxView = BoundingBoxView()
        let parentLayer = CALayer()

        XCTAssertEqual(parentLayer.sublayers?.count ?? 0, 0)

        boxView.addToLayer(parentLayer)

        XCTAssertEqual(parentLayer.sublayers?.count, 2)
        XCTAssertTrue(parentLayer.sublayers?.contains(boxView.shapeLayer) ?? false)
        XCTAssertTrue(parentLayer.sublayers?.contains(boxView.textLayer) ?? false)
    }

    func testBoundingBoxViewShow() {
        // Test BoundingBoxView show method
        let boxView = BoundingBoxView()
        let frame = CGRect(x: 10, y: 20, width: 100, height: 50)
        let label = "person 85%"
        let color = UIColor.red
        let alpha: CGFloat = 0.8

        boxView.show(frame: frame, label: label, color: color, alpha: alpha)

        // Test that layers become visible
        XCTAssertFalse(boxView.shapeLayer.isHidden)
        XCTAssertFalse(boxView.textLayer.isHidden)

        // Test shape layer properties
        XCTAssertNotNil(boxView.shapeLayer.path)
        XCTAssertEqual(boxView.shapeLayer.strokeColor, color.withAlphaComponent(alpha).cgColor)

        // Test text layer properties
        XCTAssertEqual(boxView.textLayer.string as? String, label)
        XCTAssertEqual(boxView.textLayer.backgroundColor, color.withAlphaComponent(alpha).cgColor)
        XCTAssertEqual(boxView.textLayer.foregroundColor, UIColor.white.withAlphaComponent(alpha).cgColor)
    }

    func testBoundingBoxViewHide() {
        // Test BoundingBoxView hide method
        let boxView = BoundingBoxView()

        // First show the box
        boxView.show(frame: CGRect(x: 0, y: 0, width: 50, height: 30), label: "test", color: .blue, alpha: 1.0)
        XCTAssertFalse(boxView.shapeLayer.isHidden)
        XCTAssertFalse(boxView.textLayer.isHidden)

        // Then hide it
        boxView.hide()
        XCTAssertTrue(boxView.shapeLayer.isHidden)
        XCTAssertTrue(boxView.textLayer.isHidden)
    }

    func testBoundingBoxViewShowWithDifferentColors() {
        // Test BoundingBoxView show with different colors
        let boxView = BoundingBoxView()
        let frame = CGRect(x: 0, y: 0, width: 80, height: 60)
        let colors = [UIColor.red, UIColor.blue, UIColor.green, UIColor.yellow, UIColor.purple]

        for (index, color) in colors.enumerated() {
            let alpha: CGFloat = CGFloat(index + 1) * 0.2
            boxView.show(frame: frame, label: "test \(index)", color: color, alpha: alpha)

            XCTAssertEqual(boxView.shapeLayer.strokeColor, color.withAlphaComponent(alpha).cgColor)
            XCTAssertEqual(boxView.textLayer.backgroundColor, color.withAlphaComponent(alpha).cgColor)
        }
    }

    func testBoundingBoxViewShowWithDifferentFrames() {
        // Test BoundingBoxView show with different frame sizes
        let boxView = BoundingBoxView()
        let frames = [
            CGRect(x: 0, y: 0, width: 50, height: 30),
            CGRect(x: 10, y: 20, width: 100, height: 80),
            CGRect(x: 5, y: 5, width: 200, height: 150)
        ]
        
        for frame in frames {
            boxView.show(frame: frame, label: "object", color: .red, alpha: 0.7)
            
            if let path = boxView.shapeLayer.path {
                let pathBounds = path.boundingBox
                XCTAssertEqual(pathBounds.origin.x, frame.origin.x, accuracy: 1.0)
                XCTAssertEqual(pathBounds.origin.y, frame.origin.y, accuracy: 1.0)
                XCTAssertEqual(pathBounds.size.width, frame.size.width, accuracy: 1.0)
                XCTAssertEqual(pathBounds.size.height, frame.size.height, accuracy: 1.0)
            }
        }
    }
}

// MARK: - Tests for BoundingBoxInfo

class BoundingBoxInfoTests: XCTestCase {
    
    func testBoundingBoxInfoCreation() {
        // Test BoundingBoxInfo struct creation
        let rect = CGRect(x: 10, y: 20, width: 100, height: 50)
        let strokeColor = UIColor.blue
        let strokeWidth: CGFloat = 3.0
        let cornerRadius: CGFloat = 8.0
        let alpha: CGFloat = 0.9
        let labelText = "car 92%"
        let labelFont = UIFont.systemFont(ofSize: 16)
        let labelTextColor = UIColor.white
        let labelBackgroundColor = UIColor.red
        let isHidden = false
        
        let info = BoundingBoxInfo(
            rect: rect,
            strokeColor: strokeColor,
            strokeWidth: strokeWidth,
            cornerRadius: cornerRadius,
            alpha: alpha,
            labelText: labelText,
            labelFont: labelFont,
            labelTextColor: labelTextColor,
            labelBackgroundColor: labelBackgroundColor,
            isHidden: isHidden
        )
        
        XCTAssertEqual(info.rect, rect)
        XCTAssertEqual(info.strokeColor, strokeColor)
        XCTAssertEqual(info.strokeWidth, strokeWidth)
        XCTAssertEqual(info.cornerRadius, cornerRadius)
        XCTAssertEqual(info.alpha, alpha)
        XCTAssertEqual(info.labelText, labelText)
        XCTAssertEqual(info.labelFont, labelFont)
        XCTAssertEqual(info.labelTextColor, labelTextColor)
        XCTAssertEqual(info.labelBackgroundColor, labelBackgroundColor)
        XCTAssertEqual(info.isHidden, isHidden)
    }
    
    @MainActor
    func testCreateBoxViewFromInfo() {
        // Test createBoxView utility function
        let info = BoundingBoxInfo(
            rect: CGRect(x: 0, y: 0, width: 120, height: 80),
            strokeColor: .green,
            strokeWidth: 2.5,
            cornerRadius: 6.0,
            alpha: 0.8,
            labelText: "bicycle 76%",
            labelFont: UIFont.systemFont(ofSize: 14),
            labelTextColor: .white,
            labelBackgroundColor: .green,
            isHidden: false
        )
        
        let boxView = createBoxView(from: info)
        
        XCTAssertNotNil(boxView)
        XCTAssertEqual(boxView.layer.borderColor, info.strokeColor.withAlphaComponent(info.alpha).cgColor)
        XCTAssertEqual(boxView.layer.borderWidth, info.strokeWidth)
        XCTAssertEqual(boxView.layer.cornerRadius, info.cornerRadius)
        XCTAssertEqual(boxView.backgroundColor, .clear)
        
        // Check that label was added
        XCTAssertGreaterThan(boxView.subviews.count, 0)
        let label = boxView.subviews.first as? UILabel
        XCTAssertNotNil(label)
        XCTAssertEqual(label?.text, info.labelText)
    }
    
    @MainActor
    func testMakeBoundingBoxInfosFromVisibleBoxes() {
        // Test makeBoundingBoxInfos with visible boxes
        let boxView1 = BoundingBoxView()
        let boxView2 = BoundingBoxView()
        
        // Show first box
        boxView1.show(
            frame: CGRect(x: 10, y: 10, width: 50, height: 40),
            label: "person 90%",
            color: .red,
            alpha: 0.8
        )
        
        // Show second box
        boxView2.show(
            frame: CGRect(x: 20, y: 30, width: 60, height: 45),
            label: "car 85%",
            color: .blue,
            alpha: 0.7
        )
        
        let infos = makeBoundingBoxInfos(from: [boxView1, boxView2])
        
        XCTAssertEqual(infos.count, 2)
        
        // Test first box info
        let info1 = infos[0]
        XCTAssertEqual(info1.labelText, "person 90%")
        XCTAssertFalse(info1.isHidden)
        XCTAssertEqual(info1.strokeWidth, 4.0) // BoundingBoxView default
        XCTAssertEqual(info1.cornerRadius, 6.0) // BoundingBoxView default
        
        // Test second box info
        let info2 = infos[1]
        XCTAssertEqual(info2.labelText, "car 85%")
        XCTAssertFalse(info2.isHidden)
    }
    
    @MainActor
    func testMakeBoundingBoxInfosFromHiddenBoxes() {
        // Test makeBoundingBoxInfos with hidden boxes
        let boxView1 = BoundingBoxView()
        let boxView2 = BoundingBoxView()
        
        // Show first box, then hide it
        boxView1.show(frame: CGRect(x: 0, y: 0, width: 50, height: 30), label: "test", color: .red, alpha: 1.0)
        boxView1.hide()
        
        // Leave second box hidden (never shown)
        
        let infos = makeBoundingBoxInfos(from: [boxView1, boxView2])
        
        // Should return empty array since all boxes are hidden
        XCTAssertEqual(infos.count, 0)
    }
    
    @MainActor
    func testMakeBoundingBoxInfosFromMixedBoxes() {
        // Test makeBoundingBoxInfos with mix of visible and hidden boxes
        let boxView1 = BoundingBoxView()
        let boxView2 = BoundingBoxView()
        let boxView3 = BoundingBoxView()
        
        // Show first box
        boxView1.show(frame: CGRect(x: 0, y: 0, width: 40, height: 30), label: "visible", color: .green, alpha: 1.0)
        
        // Hide second box
        boxView2.hide()
        
        // Show third box
        boxView3.show(frame: CGRect(x: 50, y: 50, width: 60, height: 40), label: "also visible", color: .purple, alpha: 0.9)
        
        let infos = makeBoundingBoxInfos(from: [boxView1, boxView2, boxView3])
        
        // Should return info for 2 visible boxes
        XCTAssertEqual(infos.count, 2)
        XCTAssertEqual(infos[0].labelText, "visible")
        XCTAssertEqual(infos[1].labelText, "also visible")
    }
    
    @MainActor
    func testMakeBoundingBoxInfosFromEmptyArray() {
        // Test makeBoundingBoxInfos with empty array
        let infos = makeBoundingBoxInfos(from: [])
        
        XCTAssertEqual(infos.count, 0)
    }
}
