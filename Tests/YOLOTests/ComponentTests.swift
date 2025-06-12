// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import UIKit
import XCTest
import AVFoundation

@testable import YOLO

/// Tests for UI components and visualization functions
@MainActor
class YOLOUITests: XCTestCase {
    
    // MARK: - YOLOView Tests
    
    func testYOLOViewInitialization() {
        let frame = CGRect(x: 0, y: 0, width: 300, height: 400)
        
        if let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources") {
            let yoloView = YOLOView(frame: frame, modelPathOrName: modelURL.path, task: .detect)
            
            XCTAssertEqual(yoloView.frame, frame)
            XCTAssertNotNil(yoloView.boundingBoxViews)
            XCTAssertEqual(yoloView.boundingBoxViews.count, yoloView.maxBoundingBoxViews)
        }
    }
    
    func testYOLOViewTaskSetting() {
        let frame = CGRect(x: 0, y: 0, width: 300, height: 400)
        
        if let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources") {
            let yoloView = YOLOView(frame: frame, modelPathOrName: modelURL.path, task: .segment)
            XCTAssertEqual(yoloView.task, .segment)
        }
    }
    
    func testYOLOViewSliderControls() {
        let frame = CGRect(x: 0, y: 0, width: 300, height: 400)
        
        if let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources") {
            let yoloView = YOLOView(frame: frame, modelPathOrName: modelURL.path, task: .detect)
            
            XCTAssertEqual(yoloView.sliderConf.value, 0.25, accuracy: 0.01)
            XCTAssertEqual(yoloView.sliderIoU.value, 0.45, accuracy: 0.01)
            XCTAssertEqual(yoloView.sliderNumItems.value, 30, accuracy: 0.01)
        }
    }
    
    // MARK: - YOLOCamera Tests
    
    func testYOLOCameraInitialization() {
        if let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources") {
            let camera = YOLOCamera(modelPathOrName: modelURL.path, task: .detect, cameraPosition: .back)
            
            XCTAssertEqual(camera.task, .detect)
            XCTAssertEqual(camera.cameraPosition, .back)
            XCTAssertEqual(camera.modelPathOrName, modelURL.path)
        }
    }
    
    func testYOLOCameraFrontCamera() {
        if let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources") {
            let camera = YOLOCamera(modelPathOrName: modelURL.path, task: .classify, cameraPosition: .front)
            
            XCTAssertEqual(camera.cameraPosition, .front)
            XCTAssertEqual(camera.task, .classify)
        }
    }
    
    // MARK: - BoundingBox Utility Tests
    
    func testBoundingBoxInfoCreation() {
        let info = BoundingBoxInfo(
            rect: CGRect(x: 10, y: 20, width: 100, height: 50),
            strokeColor: .red,
            strokeWidth: 2.0,
            cornerRadius: 5.0,
            alpha: 0.8,
            labelText: "Test Label",
            labelFont: UIFont.systemFont(ofSize: 14),
            labelTextColor: .white,
            labelBackgroundColor: .blue,
            isHidden: false
        )
        
        XCTAssertEqual(info.rect.width, 100)
        XCTAssertEqual(info.strokeWidth, 2.0)
        XCTAssertEqual(info.labelText, "Test Label")
        XCTAssertFalse(info.isHidden)
    }
    
    func testCreateBoxViewFromInfo() {
        let info = BoundingBoxInfo(
            rect: CGRect(x: 0, y: 0, width: 100, height: 50),
            strokeColor: .green,
            strokeWidth: 3.0,
            cornerRadius: 8.0,
            alpha: 0.9,
            labelText: "Object",
            labelFont: UIFont.systemFont(ofSize: 16),
            labelTextColor: .black,
            labelBackgroundColor: .yellow,
            isHidden: false
        )
        
        let boxView = createBoxView(from: info)
        
        XCTAssertNotNil(boxView)
        XCTAssertEqual(boxView.layer.cornerRadius, 8.0)
        XCTAssertEqual(boxView.layer.borderWidth, 3.0)
    }
    
    func testMakeBoundingBoxInfos() {
        let boundingBoxViews = [BoundingBoxView(), BoundingBoxView()]
        
        // Show one box
        boundingBoxViews[0].show(
            frame: CGRect(x: 10, y: 10, width: 50, height: 50),
            label: "Test",
            color: .red,
            alpha: 0.8
        )
        
        let infos = makeBoundingBoxInfos(from: boundingBoxViews)
        
        XCTAssertEqual(infos.count, 1) // Only one visible box
        XCTAssertEqual(infos[0].labelText, "Test")
    }
    
    // MARK: - Visualization Function Tests
    
    func testDrawYOLODetections() {
        let size = CGSize(width: 100, height: 100)
        let ciImage = createTestCIImage(size: size, color: .blue)
        
        let boxes = [Box(
            index: 0,
            cls: "test",
            conf: 0.9,
            xywh: CGRect(x: 10, y: 10, width: 30, height: 30),
            xywhn: CGRect(x: 0.1, y: 0.1, width: 0.3, height: 0.3)
        )]
        
        let result = YOLOResult(orig_shape: size, boxes: boxes, speed: 0.1, names: ["test"])
        let drawnImage = drawYOLODetections(on: ciImage, result: result)
        
        XCTAssertNotNil(drawnImage)
        XCTAssertGreaterThan(drawnImage.size.width, 0)
        XCTAssertGreaterThan(drawnImage.size.height, 0)
    }
    
    func testDrawYOLOClassifications() {
        let size = CGSize(width: 200, height: 200)
        let ciImage = createTestCIImage(size: size, color: .green)
        
        let probs = Probs(
            top1: "cat",
            top5: ["cat", "dog", "bird", "fish", "mouse"],
            top1Conf: 0.95,
            top5Confs: [0.95, 0.8, 0.6, 0.4, 0.2]
        )
        
        let result = YOLOResult(
            orig_shape: size,
            boxes: [],
            probs: probs,
            speed: 0.05,
            names: ["cat", "dog", "bird", "fish", "mouse"]
        )
        
        let drawnImage = drawYOLOClassifications(on: ciImage, result: result)
        
        XCTAssertNotNil(drawnImage)
        XCTAssertEqual(drawnImage.size.width, size.width)
        XCTAssertEqual(drawnImage.size.height, size.height)
    }
    
    // MARK: - Color Utility Tests
    
    func testUIColorRGBComponents() {
        let redColor = UIColor.red
        let components = redColor.toRGBComponents()
        
        XCTAssertNotNil(components)
        XCTAssertEqual(components?.red, UInt8(255))
        XCTAssertEqual(components?.green, UInt8(0))
        XCTAssertEqual(components?.blue, UInt8(0))
    }
    
    func testUIColorCustomRGBComponents() {
        let customColor = UIColor(red: 0.5, green: 0.3, blue: 0.8, alpha: 1.0)
        let components = customColor.toRGBComponents()
        
        XCTAssertNotNil(components)
        // Use range checking instead of accuracy parameter for UInt8
        XCTAssertTrue(abs(Int(components?.red ?? 0) - 127) <= 2)
        XCTAssertTrue(abs(Int(components?.green ?? 0) - 76) <= 2)  
        XCTAssertTrue(abs(Int(components?.blue ?? 0) - 204) <= 2)
    }
    
    // MARK: - Process String Tests
    
    func testProcessString() {
        XCTAssertEqual(processString("yolo11n"), "YOLO11n")
        XCTAssertEqual(processString("model-obb"), "Model-OBB")
        XCTAssertEqual(processString("custom_model"), "Custom_model")
        XCTAssertEqual(processString(""), "")
    }
    
    // MARK: - Layout Tests
    
    func testYOLOViewLayoutSubviews() {
        let frame = CGRect(x: 0, y: 0, width: 400, height: 600)
        
        if let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources") {
            let yoloView = YOLOView(frame: frame, modelPathOrName: modelURL.path, task: .detect)
            
            // Trigger layout
            yoloView.layoutSubviews()
            
            // Verify UI elements are positioned
            XCTAssertFalse(yoloView.labelName.frame.isEmpty)
            XCTAssertFalse(yoloView.labelFPS.frame.isEmpty)
            XCTAssertFalse(yoloView.toolbar.frame.isEmpty)
        }
    }
    
    // MARK: - Delegate Tests
    
    func testYOLOViewDelegate() {
        let frame = CGRect(x: 0, y: 0, width: 300, height: 400)
        
        if let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources") {
            let yoloView = YOLOView(frame: frame, modelPathOrName: modelURL.path, task: .detect)
            let delegate = MockYOLOViewDelegate()
            
            yoloView.delegate = delegate
            
            // Test delegate assignment
            XCTAssertNotNil(yoloView.delegate)
            XCTAssertTrue(yoloView.delegate === delegate)
        }
    }
    
    // MARK: - Helper Methods
    
    private func createTestCIImage(size: CGSize, color: UIColor) -> CIImage {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        color.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return CIImage(image: image!) ?? CIImage()
    }
}

// MARK: - Mock Delegate

class MockYOLOViewDelegate: YOLOViewDelegate {
    var lastFPS: Double = 0
    var lastInferenceTime: Double = 0
    var lastResult: YOLOResult?
    
    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        lastFPS = fps
        lastInferenceTime = inferenceTime
    }
    
    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        lastResult = result
    }
}
