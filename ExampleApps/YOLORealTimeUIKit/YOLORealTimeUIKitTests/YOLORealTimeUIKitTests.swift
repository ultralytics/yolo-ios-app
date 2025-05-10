// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing unit tests for the real-time UIKit example.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import AVFoundation
import Testing
import UIKit
import Vision
import XCTest

@testable import YOLO
@testable import YOLORealTimeUIKit

/// Unit tests for the YOLO RealTime UIKit example application.
///
/// This test suite verifies the functionality of the real-time object detection application
/// that uses UIKit and the YOLO framework. It contains tests that validate the core features
/// of the app, including model initialization, camera configuration, and UI interactions.
///
/// - Note: These tests require the application to be built with testing enabled.
/// - Important: Some tests may require the YOLO11 detection model to be available.
struct YOLORealTimeUIKitTests {

  // Flag to skip model-dependent tests if model is not available
  static let SKIP_MODEL_TESTS = true

  /// Tests that the view controller initializes correctly.
  @Test func testViewControllerInitialization() async throws {
    let viewController = ViewController()
    _ = viewController.view  // Force view to load

    #expect(viewController.view != nil)
    #expect(viewController.yoloView != nil)
  }

  /// Tests the YOLOView configuration.
  @Test func testYOLOViewConfiguration() async throws {
    if YOLORealTimeUIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping testYOLOViewConfiguration as model is not prepared")
      return
    }

    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    let yoloView = await YOLOView(frame: frame, modelPathOrName: "yolo11n", task: .detect)

    // Allow some time for initialization to complete
    try await Task.sleep(for: .seconds(0.5))

    #expect(await yoloView.task == .detect)
    #expect(yoloView.frame == frame)

    // Test that the video capture session is properly initialized
    if let videoCapture = Mirror(reflecting: yoloView).children.first(where: {
      $0.label == "videoCapture"
    })?.value as? VideoCapture {
      #expect(videoCapture.captureSession != nil)
    } else {
      XCTFail("Could not access videoCapture property")
    }
  }

  /// Tests UI control initialization and functionality.
  @Test func testUIControls() async throws {
    if YOLORealTimeUIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping testUIControls as model is not prepared")
      return
    }

    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    let yoloView = await YOLOView(frame: frame, modelPathOrName: "yolo11n", task: .detect)

    // Allow some time for initialization to complete
    try await Task.sleep(for: .seconds(0.5))

    // Test slider initialization
    await #expect(yoloView.sliderConf.minimumValue == 0)
    await #expect(yoloView.sliderConf.maximumValue == 1)
    await #expect(yoloView.sliderConf.value == 0.25)  // Default confidence threshold

    await #expect(yoloView.sliderIoU.minimumValue == 0)
    await #expect(yoloView.sliderIoU.maximumValue == 1)
    await #expect(yoloView.sliderIoU.value == 0.45)  // Default IoU threshold

    await #expect(yoloView.sliderNumItems.minimumValue == 0)
    await #expect(yoloView.sliderNumItems.maximumValue == 100)
    await #expect(yoloView.sliderNumItems.value == 30)  // Default number of items

    // Test buttons
    await #expect(yoloView.playButton != nil)
    await #expect(yoloView.pauseButton != nil)
    await #expect(yoloView.switchCameraButton != nil)

    // Test labels
    await #expect(yoloView.labelName != nil)
    await #expect(yoloView.labelFPS != nil)
    await #expect(yoloView.labelSliderConf != nil)
    await #expect(yoloView.labelSliderIoU != nil)
    await #expect(yoloView.labelSliderNumItems != nil)

    // Verify label text
    await #expect(yoloView.labelSliderConf.text == "0.25 Confidence Threshold")
    await #expect(yoloView.labelSliderIoU.text == "0.45 IoU Threshold")
  }

  /// Tests the bounding box view initialization.
  @Test func testBoundingBoxViews() async throws {
    // This test requires a valid model as YOLOView fatally crashes with invalid models
    if YOLORealTimeUIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping testBoundingBoxViews as model is not prepared")
      return
    }

    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    let yoloView = await YOLOView(frame: frame, modelPathOrName: "yolo11n", task: .detect)

    // Allow some time for initialization to complete
    try await Task.sleep(for: .seconds(0.5))

    // Access the bounding box views via reflection
    if let boundingBoxViews = Mirror(reflecting: yoloView).children.first(where: {
      $0.label == "boundingBoxViews"
    })?.value as? [BoundingBoxView] {
      // Verify that bounding box views are initialized
      #expect(!boundingBoxViews.isEmpty)

      // Check the maximum number of bounding box views
      if let maxBoundingBoxViews = Mirror(reflecting: yoloView).children.first(where: {
        $0.label == "maxBoundingBoxViews"
      })?.value as? Int {
        #expect(maxBoundingBoxViews == 100)
        #expect(boundingBoxViews.count == maxBoundingBoxViews)
      } else {
        XCTFail("Could not access maxBoundingBoxViews property")
      }
    } else {
      XCTFail("Could not access boundingBoxViews property")
    }
  }

  /// Documentation test for model error handling limitations
  @Test func testModelErrorHandlingDocumentation() async throws {
    // Note: We cannot directly test invalid model paths because YOLOView calls fatalError
    // when a model is not found (in Sources/YOLO/YOLOView.swift:193)

    // Document this design limitation for future improvement
    let improvement = """
      YOLOView should be refactored to handle missing models gracefully by:
      1. Using completion handlers with Result type instead of fatalError
      2. Using Swift's throwing mechanisms
      3. Using optional values or fallbacks

      Current implementation in YOLOView.swift line 193 uses:
      guard let unwrappedModelURL = modelURL else {
        let error = PredictorError.modelFileNotFound
        fatalError(error.localizedDescription)
      }
      """

    // This is just a documentation test
    #expect(true, "This test documents a design limitation")
  }

  /// Tests the basic button functionality of play/pause controls.
  /// - Note: This test can only be run when a valid model is available.
  @Test func testPlayPauseButtonsFunctionality() async throws {
    // Skip this test if no model is available
    if YOLORealTimeUIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping button functionality test as model is not prepared")
      return
    }

    // Create YOLOView with valid model
    let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
    let yoloView = await YOLOView(frame: frame, modelPathOrName: "yolo11n", task: .detect)

    // Allow initialization to complete
    try await Task.sleep(for: .seconds(0.5))

    // Verify default button states
    #expect(await yoloView.playButton.isEnabled == false)
    #expect(await yoloView.pauseButton.isEnabled == true)

    // Test pause button action
    await yoloView.pauseButton.sendActions(for: .touchUpInside)
    try await Task.sleep(for: .seconds(0.1))

    // Verify button states changed appropriately
    #expect(await yoloView.playButton.isEnabled == true)
    #expect(await yoloView.pauseButton.isEnabled == false)

    // Test play button action
    await yoloView.playButton.sendActions(for: .touchUpInside)
    try await Task.sleep(for: .seconds(0.1))

    // Verify button states returned to original state
    #expect(await yoloView.playButton.isEnabled == false)
    #expect(await yoloView.pauseButton.isEnabled == true)
  }
}

// getPrivateProperty function is no longer used, functionality replaced with direct Mirror access
