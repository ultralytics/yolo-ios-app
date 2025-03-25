//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing unit tests for the single image UIKit example.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import XCTest
import UIKit
import Testing
import PhotosUI
import CoreImage

@testable import YOLO_Single_Image_UIKit
@testable import YOLO

/// Unit tests for the YOLO Single Image UIKit example application.
///
/// This test suite verifies the functionality of the single image processing application
/// that uses UIKit and the YOLO framework. It contains tests that validate the core features
/// of the app, including model initialization, image processing, and UI interactions.
///
/// - Note: These tests require the application to be built with testing enabled.
/// - Important: Some tests may require the YOLO11 segmentation model to be available.
struct YOLO_Single_Image_UIKitTests {

  // Flag to skip model-dependent tests if model is not available
  static let SKIP_MODEL_TESTS = true
  
  /// Tests that the view controller initializes correctly.
  @Test func testViewControllerInitialization() async throws {
    let viewController = ViewController()
    _ = viewController.view // Force view to load
    
    #expect(viewController.view != nil)
  }
  
  /// Tests the image orientation correction functionality.
  @Test func testImageOrientationCorrection() async throws {
    let viewController = ViewController()
    
    // Create test images with different orientations
    let size = CGSize(width: 100, height: 100)
    UIGraphicsBeginImageContext(size)
    UIColor.red.setFill()
    UIRectFill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
    let originalImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()
    
    // Test with default orientation
    let correctedImage1 = viewController.getCorrectOrientationUIImage(uiImage: originalImage)
    #expect(correctedImage1.size.width == originalImage.size.width)
    #expect(correctedImage1.size.height == originalImage.size.height)
    
    // Create image with orientation = .down (1)
    let imageDown = UIImage(cgImage: originalImage.cgImage!, scale: 1.0, orientation: .down)
    let correctedImageDown = viewController.getCorrectOrientationUIImage(uiImage: imageDown)
    #expect(correctedImageDown.imageOrientation != imageDown.imageOrientation)
    
    // Create image with orientation = .left (3)
    let imageLeft = UIImage(cgImage: originalImage.cgImage!, scale: 1.0, orientation: .left)
    let correctedImageLeft = viewController.getCorrectOrientationUIImage(uiImage: imageLeft)
    #expect(correctedImageLeft.imageOrientation != imageLeft.imageOrientation)
  }
  
  /// Tests the UI setup functionality.
  @Test func testUISetup() async throws {
    if YOLO_Single_Image_UIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping testUISetup as model is not prepared")
      return
    }
    
    let viewController = ViewController()
    
    // Simulate the model loading callback by directly calling setupView
    viewController.model = YOLO("yolo11x-seg", task: .segment) { _ in }
    viewController.setupView()
    
    #expect(viewController.imageView != nil)
    #expect(viewController.pickButton != nil)
    #expect(viewController.pickButton.title(for: .normal) == "Pick Image")
  }
  
  /// Tests the photo picker functionality.
  @Test func testPhotoPickerPresentation() async throws {
    let viewController = ViewController()
    _ = viewController.view
    
    // Ensure the picker can be initialized
    var config = PHPickerConfiguration()
    config.selectionLimit = 1
    let picker = PHPickerViewController(configuration: config)
    picker.delegate = viewController
    
    #expect(picker != nil)
  }
  
  /// Tests the model inference with a test image.
  @Test func testModelInference() async throws {
    if YOLO_Single_Image_UIKitTests.SKIP_MODEL_TESTS {
      #warning("Skipping testModelInference as model is not prepared")
      return
    }
    
    // This test requires a loaded model
    var didInitializeModel = false
    let expectation = XCTestExpectation(description: "Model initialization")
    
    let viewController = ViewController()
    viewController.model = YOLO("yolo11x-seg", task: .segment) { result in
      if case .success = result {
        didInitializeModel = true
        expectation.fulfill()
      }
    }
    
    await fulfillment(of: [expectation], timeout: 5.0)
    #expect(didInitializeModel)
    
    if didInitializeModel {
      // Create a test image
      let size = CGSize(width: 640, height: 640)
      UIGraphicsBeginImageContext(size)
      UIColor.white.setFill()
      UIRectFill(CGRect(origin: .zero, size: size))
      UIColor.black.setFill()
      UIRectFill(CGRect(x: 200, y: 200, width: 240, height: 240))
      let testImage = UIGraphicsGetImageFromCurrentImageContext()!
      UIGraphicsEndImageContext()
      
      // Perform inference
      let result = viewController.model(testImage)
      
      // Verify basic result properties
      #expect(result.boxes != nil)
      #expect(result.orig_shape.width == testImage.size.width)
      #expect(result.orig_shape.height == testImage.size.height)
    }
  }
}