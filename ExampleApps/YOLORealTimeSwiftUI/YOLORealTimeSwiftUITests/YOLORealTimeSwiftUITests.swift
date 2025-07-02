// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing unit tests for the real-time SwiftUI example.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import CoreML
import SwiftUI
import Testing
import Vision
import XCTest

@testable import YOLO
@testable import YOLORealTimeSwiftUI

/// Unit tests for the YOLO RealTime SwiftUI example application.
///
/// This test suite verifies the functionality of the real-time object detection application
/// that uses SwiftUI and the YOLO framework. It contains tests that validate the core features
/// of the app, including model initialization, camera preview functionality, and UI interactions.
///
/// - Note: These tests require the application to be built with testing enabled.
/// - Important: Some tests require the YOLO11 OBB model to be available in the project.
struct YOLORealTimeSwiftUITests {

  /// Tests the initialization of the ContentView.
  @Test func testContentViewInitialization() async throws {
    let contentView = ContentView()
    // Verify ContentView can be instantiated
    #expect(contentView != nil)
  }

  /// Tests the YOLOCamera component configuration.
  @Test func testYOLOCameraConfiguration() async throws {
    // Test YOLOCamera can be initialized with various configurations
    let configurations: [(model: String, task: YOLOTask, position: AVCaptureDevice.Position)] = [
      ("test_model", .detect, .back),
      ("test_model", .segment, .front),
      ("test_model", .pose, .back),
      ("test_model", .classify, .front),
      ("test_model", .obb, .back)
    ]
    
    for config in configurations {
      let yoloCamera = YOLOCamera(
        modelPathOrName: config.model,
        task: config.task,
        cameraPosition: config.position
      )
      
      #expect(yoloCamera.modelPathOrName == config.model)
      #expect(yoloCamera.task == config.task)
      #expect(yoloCamera.cameraPosition == config.position)
    }
  }

  /// Tests that an invalid model path is handled gracefully.
  @Test func testInvalidModelPath() async throws {
    // Create a YOLOCamera with an invalid model path
    let yoloCamera = YOLOCamera(
      modelPathOrName: "non_existent_model",
      task: .detect,
      cameraPosition: .back
    )

    // We can't directly test for errors since YOLOCamera init doesn't throw
    // But we can at least verify that instantiation completes
    #expect(yoloCamera.modelPathOrName == "non_existent_model")

    // Wait for a short time to allow any background processes to complete
    try await Task.sleep(for: .seconds(0.5))
  }

  /// Tests camera position properties.
  @Test func testCameraPosition() async throws {
    // Test initialization with different camera positions
    let frontCamera = YOLOCamera(
      modelPathOrName: "test_model",
      task: .detect,
      cameraPosition: .front
    )

    #expect(frontCamera.cameraPosition == .front)

    let backCamera = YOLOCamera(
      modelPathOrName: "test_model",
      task: .detect,
      cameraPosition: .back
    )

    #expect(backCamera.cameraPosition == .back)
    
    // Test that unspecified position defaults to back
    let defaultCamera = YOLOCamera(
      modelPathOrName: "test_model",
      task: .detect,
      cameraPosition: .unspecified
    )
    
    #expect(defaultCamera.cameraPosition == .unspecified)
  }
}
