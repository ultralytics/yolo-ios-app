// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import UIKit
import AVFoundation
@testable import YOLO

/// Minimal tests for YOLOView core functionality
class YOLOViewTests: XCTestCase {
    
    @MainActor
    func testYOLOViewInitialization() {
        // Test YOLOView initialization with frame and model
        let frame = CGRect(x: 0, y: 0, width: 400, height: 600)
        let yoloView = createTestYOLOView(frame: frame, task: .detect)
        
        XCTAssertNotNil(yoloView)
        XCTAssertEqual(yoloView.frame, frame)
        XCTAssertEqual(yoloView.task, .detect)
        XCTAssertEqual(yoloView.maxBoundingBoxViews, 100)
        XCTAssertEqual(yoloView.boundingBoxViews.count, 100)
    }
    
    @MainActor
    func testYOLOViewInitializationWithDifferentTasks() {
        // Test YOLOView initialization with different tasks
        let frame = CGRect(x: 0, y: 0, width: 300, height: 400)
        let tasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]
        
        for task in tasks {
            let yoloView = createTestYOLOView(frame: frame, task: task)
            XCTAssertEqual(yoloView.task, task)
        }
    }
    
    @MainActor
    func testYOLOViewUIElementsExist() {
        // Test that YOLOView UI elements are properly initialized
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        
        XCTAssertNotNil(yoloView.sliderNumItems)
        XCTAssertNotNil(yoloView.labelSliderNumItems)
        XCTAssertNotNil(yoloView.sliderConf)
        XCTAssertNotNil(yoloView.labelSliderConf)
        XCTAssertNotNil(yoloView.sliderIoU)
        XCTAssertNotNil(yoloView.labelSliderIoU)
        XCTAssertNotNil(yoloView.labelName)
        XCTAssertNotNil(yoloView.labelFPS)
        XCTAssertNotNil(yoloView.labelZoom)
        XCTAssertNotNil(yoloView.activityIndicator)
        XCTAssertNotNil(yoloView.playButton)
        XCTAssertNotNil(yoloView.pauseButton)
        XCTAssertNotNil(yoloView.switchCameraButton)
        XCTAssertNotNil(yoloView.toolbar)
    }
    
    @MainActor
    func testYOLOViewSliderDefaults() {
        // Test YOLOView slider default values
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        
        // Test slider default values
        XCTAssertEqual(yoloView.sliderNumItems.value, 30)
        XCTAssertEqual(yoloView.sliderNumItems.minimumValue, 0)
        XCTAssertEqual(yoloView.sliderNumItems.maximumValue, 100)
        
        XCTAssertEqual(yoloView.sliderConf.value, 0.25, accuracy: 0.001)
        XCTAssertEqual(yoloView.sliderConf.minimumValue, 0)
        XCTAssertEqual(yoloView.sliderConf.maximumValue, 1)
        
        XCTAssertEqual(yoloView.sliderIoU.value, 0.45, accuracy: 0.001)
        XCTAssertEqual(yoloView.sliderIoU.minimumValue, 0)
        XCTAssertEqual(yoloView.sliderIoU.maximumValue, 1)
    }
    
    @MainActor
    func testYOLOViewButtonStates() {
        // Test YOLOView button initial states
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        
        XCTAssertFalse(yoloView.playButton.isEnabled)
        XCTAssertTrue(yoloView.pauseButton.isEnabled)
    }
    
    @MainActor
    func testYOLOViewZoomProperties() {
        // Test YOLOView zoom-related properties
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        
        XCTAssertEqual(yoloView.labelZoom.text, "1.00x")
        XCTAssertEqual(yoloView.labelZoom.textColor, .white)
    }
    
    @MainActor
    func testYOLOViewBoundingBoxViewsSetup() {
        // Test YOLOView bounding box views setup
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        
        XCTAssertEqual(yoloView.boundingBoxViews.count, yoloView.maxBoundingBoxViews)
        
        // All bounding box views should be hidden initially
        for boxView in yoloView.boundingBoxViews {
            XCTAssertTrue(boxView.shapeLayer.isHidden)
            XCTAssertTrue(boxView.textLayer.isHidden)
        }
    }
    
    @MainActor
    func testYOLOViewLayoutSubviews() {
        // Test YOLOView layoutSubviews method
        let frame = CGRect(x: 0, y: 0, width: 400, height: 600)
        let yoloView = createTestYOLOView(frame: frame, task: .detect)
        
        // Trigger layout
        yoloView.layoutSubviews()
        
        // Check that UI elements have been positioned
        XCTAssertFalse(yoloView.labelName.frame.isEmpty)
        XCTAssertFalse(yoloView.labelFPS.frame.isEmpty)
        XCTAssertFalse(yoloView.toolbar.frame.isEmpty)
        XCTAssertFalse(yoloView.sliderConf.frame.isEmpty)
        XCTAssertFalse(yoloView.sliderIoU.frame.isEmpty)
        XCTAssertFalse(yoloView.sliderNumItems.frame.isEmpty)
    }
    
    @MainActor
    func testYOLOViewStop() {
        // Test YOLOView stop method
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        
        // Should not crash
        yoloView.stop()
        XCTAssertTrue(true)
    }
    
    @MainActor
    func testYOLOViewResume() {
        // Test YOLOView resume method
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        
        // Should not crash
        yoloView.resume()
        XCTAssertTrue(true)
    }
    
    @MainActor
    func testYOLOViewSetInferenceFlag() {
        // Test YOLOView setInferenceFlag method
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        
        yoloView.setInferenceFlag(ok: false)
        XCTAssertFalse(yoloView.videoCapture.inferenceOK)
        
        yoloView.setInferenceFlag(ok: true)
        XCTAssertTrue(yoloView.videoCapture.inferenceOK)
    }
    
    @MainActor
    func testYOLOViewOnDetectionCallback() {
        // Test YOLOView onDetection callback
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        let expectation = XCTestExpectation(description: "Detection callback")
        
        yoloView.onDetection = { result in
            XCTAssertNotNil(result)
            expectation.fulfill()
        }
        
        // Trigger callback manually
        let testResult = YOLOResult(orig_shape: CGSize(width: 400, height: 600), boxes: [], speed: 0.1, names: [])
        yoloView.onDetection?(testResult)
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    @MainActor
    func testYOLOViewDelegate() {
        // Test YOLOView delegate functionality
        let yoloView = createTestYOLOView(frame: CGRect(x: 0, y: 0, width: 400, height: 600), task: .detect)
        let delegate = MockYOLOViewDelegate()
        
        yoloView.delegate = delegate
        
        XCTAssertNotNil(yoloView.delegate)
        XCTAssertTrue(yoloView.delegate === delegate)
        
        // Test delegate methods
        yoloView.onInferenceTime(speed: 25.5, fps: 30.0)
        XCTAssertEqual(delegate.lastFPS, 30.0)
        XCTAssertEqual(delegate.lastInferenceTime, 25.5)
        
        let testResult = YOLOResult(orig_shape: CGSize(width: 100, height: 100), boxes: [], speed: 0.1, names: [])
        yoloView.onPredict(result: testResult)
        XCTAssertNotNil(delegate.lastResult)
    }
    
    // MARK: - Helper Methods
    
    @MainActor
    private func createTestYOLOView(frame: CGRect, task: YOLOTask) -> YOLOView {
        // Create a YOLOView without actually loading a model
        let yoloView = YOLOView.__allocating_init(frame: frame, modelPathOrName: "test_model", task: task)
        return yoloView
    }
}

// MARK: - Tests for Utility Functions

class YOLOViewUtilityTests: XCTestCase {
    
    func testProcessString() {
        // Test processString utility function
        XCTAssertEqual(processString("yolo11n"), "YOLO11n")
        XCTAssertEqual(processString("yolov8s"), "YOLOv8s")
        XCTAssertEqual(processString("model-obb"), "Model-OBB")
        XCTAssertEqual(processString("custom_model"), "Custom_model")
        XCTAssertEqual(processString(""), "")
        XCTAssertEqual(processString("YOLO"), "YOLO")
        XCTAssertEqual(processString("obb"), "OBB")
        XCTAssertEqual(processString("test"), "Test")
        XCTAssertEqual(processString("Test"), "Test")
        XCTAssertEqual(processString("yolo-obb-model"), "YOLO-OBB-model")
    }
    
    func testProcessStringCaseInsensitive() {
        // Test processString handles case insensitive replacements
        XCTAssertEqual(processString("YOLO11n"), "YOLO11n")
        XCTAssertEqual(processString("Yolo11n"), "YOLO11n")
        XCTAssertEqual(processString("yOlO11n"), "YOLO11n")
        XCTAssertEqual(processString("OBB"), "OBB")
        XCTAssertEqual(processString("obb"), "OBB")
        XCTAssertEqual(processString("Obb"), "OBB")
    }
    
    func testProcessStringCapitalization() {
        // Test processString capitalizes first letter
        XCTAssertEqual(processString("model"), "Model")
        XCTAssertEqual(processString("detection"), "Detection")
        XCTAssertEqual(processString("custom"), "Custom")
        XCTAssertEqual(processString("a"), "A")
    }
    
    func testProcessStringEmptyAndEdgeCases() {
        // Test processString edge cases
        XCTAssertEqual(processString(""), "")
        XCTAssertEqual(processString("y"), "Y")
        XCTAssertEqual(processString("o"), "O")
        XCTAssertEqual(processString("ALREADY_CAPS"), "ALREADY_CAPS")
    }
}

// MARK: - Extension to allow creating YOLOView instances for testing

extension YOLOView {
    @MainActor
    static func __allocating_init(frame: CGRect, modelPathOrName: String, task: YOLOTask) -> YOLOView {
        // Create instance without going through full initialization
        let yoloView = YOLOView(frame: frame)
        yoloView.task = task
        yoloView.modelName = modelPathOrName
        
        // Initialize required properties for testing
        yoloView.setUpBoundingBoxViews()
        yoloView.setupUI()
        
        return yoloView
    }
    
    // Convenience initializer for testing
    @MainActor
    convenience init(frame: CGRect) {
        self.init()
        self.frame = frame
        self.videoCapture = VideoCapture()
        self.task = .detect
        self.modelName = ""
        self.classes = []
        self.colors = [:]
        self.boundingBoxViews = []
        self.overlayLayer = CALayer()
    }
}

// MARK: - Mock Delegate for Testing

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
