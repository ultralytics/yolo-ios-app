//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing unit tests for the real-time UIKit example.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import XCTest
import UIKit
import Testing
import AVFoundation
import Vision

@testable import YOLO_RealTime_UIKit
@testable import YOLO

/// Unit tests for the YOLO RealTime UIKit example application.
///
/// This test suite verifies the functionality of the real-time object detection application
/// that uses UIKit and the YOLO framework. It contains tests that validate the core features
/// of the app, including model initialization, camera configuration, and UI interactions.
///
/// - Note: These tests require the application to be built with testing enabled.
/// - Important: Some tests may require the YOLO11 detection model to be available.
struct YOLO_RealTime_UIKitTests {

  // Flag to skip model-dependent tests if model is not available
  static let SKIP_MODEL_TESTS = true
  
  /// Tests that the view controller initializes correctly.
  @Test func testViewControllerInitialization() async throws {
    let viewController = ViewController()
    _ = viewController.view // Force view to load
    
    #expect(viewController.view != nil)
    #expect(viewController.yoloView != nil)
  }
  
  /// Tests the YOLOView configuration.
  @Test func testYOLOViewConfiguration() async throws {
    if YOLO_RealTime_UIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping testYOLOViewConfiguration as model is not prepared")
      return
    }
    
    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    let yoloView = YOLOView(frame: frame, modelPathOrName: "yolo11n", task: .detect)
    
    // Allow some time for initialization to complete
    try await Task.sleep(for: .seconds(0.5))
    
    #expect(await yoloView.task == .detect)
    #expect(yoloView.frame == frame)
    
    // Test that the video capture session is properly initialized
    if let videoCapture = Mirror(reflecting: yoloView).children.first(where: { $0.label == "videoCapture" })?.value as? VideoCapture {
      #expect(videoCapture.captureSession != nil)
    } else {
      XCTFail("Could not access videoCapture property")
    }
  }
  
  /// Tests UI control initialization and functionality.
  @Test func testUIControls() async throws {
    if YOLO_RealTime_UIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping testUIControls as model is not prepared")
      return
    }
    
    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    let yoloView = YOLOView(frame: frame, modelPathOrName: "yolo11n", task: .detect)
    
    // Allow some time for initialization to complete
    try await Task.sleep(for: .seconds(0.5))
    
    // Test slider initialization
    #expect(yoloView.sliderConf.minimumValue == 0)
    #expect(yoloView.sliderConf.maximumValue == 1)
    #expect(yoloView.sliderConf.value == 0.25) // Default confidence threshold
    
    #expect(yoloView.sliderIoU.minimumValue == 0)
    #expect(yoloView.sliderIoU.maximumValue == 1)
    #expect(yoloView.sliderIoU.value == 0.45) // Default IoU threshold
    
    #expect(yoloView.sliderNumItems.minimumValue == 0)
    #expect(yoloView.sliderNumItems.maximumValue == 100)
    #expect(yoloView.sliderNumItems.value == 30) // Default number of items
    
    // Test buttons
    #expect(yoloView.playButton != nil)
    #expect(yoloView.pauseButton != nil)
    #expect(yoloView.switchCameraButton != nil)
    
    // Test labels
    #expect(yoloView.labelName != nil)
    #expect(yoloView.labelFPS != nil)
    #expect(yoloView.labelSliderConf != nil)
    #expect(yoloView.labelSliderIoU != nil)
    #expect(yoloView.labelSliderNumItems != nil)
    
    // Verify label text
    #expect(yoloView.labelSliderConf.text == "0.25 Confidence Threshold")
    #expect(yoloView.labelSliderIoU.text == "0.45 IoU Threshold")
  }
  
  /// Tests the bounding box view initialization.
  @Test func testBoundingBoxViews() async throws {
    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    let yoloView = YOLOView(frame: frame, modelPathOrName: "non_existent_model", task: .detect)
    
    // Allow some time for initialization to complete
    try await Task.sleep(for: .seconds(0.5))
    
    // Access the bounding box views via reflection
    if let boundingBoxViews = Mirror(reflecting: yoloView).children.first(where: { $0.label == "boundingBoxViews" })?.value as? [BoundingBoxView] {
      // Verify that bounding box views are initialized
      #expect(!boundingBoxViews.isEmpty)
      
      // Check the maximum number of bounding box views
      if let maxBoundingBoxViews = Mirror(reflecting: yoloView).children.first(where: { $0.label == "maxBoundingBoxViews" })?.value as? Int {
        #expect(maxBoundingBoxViews == 100)
        #expect(boundingBoxViews.count == maxBoundingBoxViews)
      } else {
        XCTFail("Could not access maxBoundingBoxViews property")
      }
    } else {
      XCTFail("Could not access boundingBoxViews property")
    }
  }
  
  /// Tests error handling for invalid model paths.
  @Test func testInvalidModelPath() async throws {
    let expectation = XCTestExpectation(description: "Invalid model error")
    
    // Use a clearly invalid path
    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    
    // In a real scenario, we'd use a completion handler to verify error handling
    // Here we're just testing that the initialization doesn't crash
    _ = YOLOView(frame: frame, modelPathOrName: "non_existent_model", task: .detect)
    
    // Wait briefly to ensure initialization completes
    try await Task.sleep(for: .seconds(0.5))
    
    expectation.fulfill()
    await XCTestCase().wait(for: [expectation], timeout: 1.0)
  }
  
  /// Tests that play/pause button actions work correctly.
  @Test func testPlayPauseButtons() async throws {
    if YOLO_RealTime_UIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping testPlayPauseButtons as model is not prepared")
      return
    }
    
    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    let yoloView = YOLOView(frame: frame, modelPathOrName: "yolo11n", task: .detect)
    
    // Allow some time for initialization to complete
    try await Task.sleep(for: .seconds(0.5))
    
    // Test initial state (pause should be enabled, play disabled)
    #expect(yoloView.playButton.isEnabled == false)
    #expect(yoloView.pauseButton.isEnabled == true)
    
    // Simulate tapping the pause button
    yoloView.perform(Selector(("pauseTapped")))
    
    // Now play should be enabled and pause disabled
    #expect(yoloView.playButton.isEnabled == true)
    #expect(yoloView.pauseButton.isEnabled == false)
    
    // Simulate tapping the play button
    yoloView.perform(Selector(("playTapped")))
    
    // Now play should be disabled and pause enabled again
    #expect(yoloView.playButton.isEnabled == false)
    #expect(yoloView.pauseButton.isEnabled == true)
  }
}

// getPrivateProperty function is no longer used, functionality replaced with direct Mirror access