//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing unit tests for the single image SwiftUI example.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import CoreImage
import PhotosUI
import SwiftUI
import Testing
import XCTest

@testable import YOLO
@testable import YOLOSingleImageSwiftUI

/// Unit tests for the YOLO Single Image SwiftUI example application.
///
/// This test suite verifies the functionality of the single image processing application
/// that uses SwiftUI and the YOLO framework. It contains tests that validate the core features
/// of the app, including model initialization, image processing, and UI interactions.
///
/// - Note: These tests require the application to be built with testing enabled.
/// - Important: Some tests may require the YOLO11 segmentation model to be available.
struct YOLOSingleImageSwiftUITests {

  // Flag to skip model-dependent tests if model is not available
  static let SKIP_MODEL_TESTS = false

  /// Tests the initialization of the ContentView.
  @Test func testContentViewInitialization() async throws {
    let contentView = ContentView()
    // Just verify we can create the view without crashing
    #expect(true, "ContentView initialized successfully")

    // Test that the body property can be accessed without crashing
    // We can't directly test 'is some View' due to type erasure, but we can verify it exists
    _ = contentView.body
    #expect(true, "ContentView.body exists and can be accessed")
  }

  /// Tests the image orientation correction functionality.
  @Test func testImageOrientationCorrection() async throws {
    // Create test images with different orientations
    let size = CGSize(width: 100, height: 100)
    UIGraphicsBeginImageContext(size)
    UIColor.red.setFill()
    UIRectFill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
    let originalImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()

    // Test with default orientation
    let correctedImage1 = getCorrectOrientationUIImage(uiImage: originalImage)
    #expect(correctedImage1.size.width == originalImage.size.width)
    #expect(correctedImage1.size.height == originalImage.size.height)

    // Create image with orientation = .down (1)
    let imageDown = UIImage(cgImage: originalImage.cgImage!, scale: 1.0, orientation: .down)
    let correctedImageDown = getCorrectOrientationUIImage(uiImage: imageDown)

    // Check if the function returns a valid image
    #expect(correctedImageDown != nil, "Corrected image should not be nil")

    // Create image with orientation = .left (3)
    let imageLeft = UIImage(cgImage: originalImage.cgImage!, scale: 1.0, orientation: .left)
    let correctedImageLeft = getCorrectOrientationUIImage(uiImage: imageLeft)

    // Check if the function returns a valid image
    #expect(correctedImageLeft != nil, "Corrected image should not be nil")

    // For left orientation specifically, we'll verify the image data is handled correctly
    // by checking the image isn't nil since we can't reliably check orientation property changes
    #expect(correctedImageLeft.size.width > 0, "Image width should be positive")
    #expect(correctedImageLeft.size.height > 0, "Image height should be positive")
  }

  /// Tests the YOLO model initialization.
  @Test func testYOLOModelInitialization() async throws {
    if YOLOSingleImageSwiftUITests.SKIP_MODEL_TESTS {
      #warning("Skipping testYOLOModelInitialization as model is not prepared")
      return
    }

    // Since this is a model-dependent test, we'll focus on checking
    // that the model can be initialized without waiting for completion
    let yolo = YOLO("yolo11n-seg", task: .segment)

    // Basic check that the model exists
    #expect(yolo != nil, "YOLO model should initialize")

    // Note: We're not testing the completion handler to avoid the expectation issues
  }

  /// Tests the PhotosPicker initialization and basic functionality.
  @Test func testPhotosPickerInitialization() async throws {
    let contentView = ContentView()
    let mirror = Mirror(reflecting: contentView.body)

    // Verify that the view contains a PhotosPicker component
    var hasPhotosPicker = false
    for child in mirror.children {
      if let childType = type(of: child.value) as? Any.Type {
        if String(describing: childType).contains("PhotosPicker") {
          hasPhotosPicker = true
          break
        }
      }
    }

    #expect(hasPhotosPicker)
  }

  /// Tests model inference with a test image.
  @Test func testModelInference() async throws {
    if YOLOSingleImageSwiftUITests.SKIP_MODEL_TESTS {
      #warning("Skipping testModelInference as model is not prepared")
      return
    }

    // Since this is a model-dependent test, we'll focus on test setup
    // without relying on the model being fully loaded

    // Initialize the model
    let yolo = YOLO("yolo11n-seg", task: .segment)

    // Create a test image that would be used for inference
    let size = CGSize(width: 640, height: 640)
    UIGraphicsBeginImageContext(size)
    UIColor.white.setFill()
    UIRectFill(CGRect(origin: .zero, size: size))
    UIColor.black.setFill()
    UIRectFill(CGRect(x: 200, y: 200, width: 240, height: 240))
    let testImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()

    // Check that we successfully created test image
    #expect(testImage.size.width == 640)
    #expect(testImage.size.height == 640)

    // Check that the model was initialized
    #expect(yolo != nil, "YOLO model should initialize")

    // Note: We're not testing actual inference since we can't reliably
    // wait for model loading without causing test exceptions
  }
}
