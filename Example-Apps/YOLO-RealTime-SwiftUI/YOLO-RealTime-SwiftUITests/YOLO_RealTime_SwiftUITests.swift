//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing unit tests for the real-time SwiftUI example.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import XCTest
import SwiftUI
import Testing
import CoreML
import Vision

@testable import YOLO_RealTime_SwiftUI
@testable import YOLO

/// Unit tests for the YOLO RealTime SwiftUI example application.
///
/// This test suite verifies the functionality of the real-time object detection application
/// that uses SwiftUI and the YOLO framework. It contains tests that validate the core features
/// of the app, including model initialization, camera preview functionality, and UI interactions.
///
/// - Note: These tests require the application to be built with testing enabled.
/// - Important: Some tests require the YOLO11 OBB model to be available in the project.
struct YOLO_RealTime_SwiftUITests {

  // Flag to skip model-dependent tests if model is not available
  static let SKIP_MODEL_TESTS = false
  
  /// Tests the initialization of the ContentView.
  @Test func testContentViewInitialization() async throws {
    let contentView = ContentView()
    #expect(contentView.body is YOLOCamera)
  }
  
  /// Tests the YOLOCamera component configuration.
  @Test func testYOLOCameraConfiguration() async throws {
    if YOLO_RealTime_SwiftUITests.SKIP_MODEL_TESTS {
      #warning("Skipping testYOLOCameraConfiguration as model is not prepared")
      return
    }
    
    let yoloCamera = YOLOCamera(
      modelPathOrName: "yolo11n-obb",
      task: .obb,
      cameraPosition: .back
    )
    
    #expect(yoloCamera.task == .obb)
    #expect(yoloCamera.cameraPosition == .back)
    
    // Test that the YOLOCamera has a body property
    let _ = yoloCamera.body
    // If we reach here without crashing, the body is valid
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
    if YOLO_RealTime_SwiftUITests.SKIP_MODEL_TESTS {
      #warning("Skipping testCameraPosition as model is not prepared")
      return
    }
    
    // Test initialization with front camera
    let frontCamera = YOLOCamera(
      modelPathOrName: "yolo11n-obb",
      task: .obb,
      cameraPosition: .front
    )
    
    #expect(frontCamera.cameraPosition == .front)
    
    // Test initialization with back camera
    let backCamera = YOLOCamera(
      modelPathOrName: "yolo11n-obb",
      task: .obb,
      cameraPosition: .back
    )
    
    #expect(backCamera.cameraPosition == .back)
  }
}
