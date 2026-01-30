// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import UIKit
import XCTest

@testable import YOLO

/// Tests for image orientation normalization functionality.
class ImageUtilsTests: XCTestCase {

  /// Tests that normalizeImageOrientation correctly normalizes images with different orientations.
  func testNormalizeImageOrientation() {
    // Create a test image
    let size = CGSize(width: 100, height: 100)
    UIGraphicsBeginImageContext(size)
    UIColor.red.setFill()
    UIRectFill(CGRect(origin: .zero, size: size))
    let originalImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()

    // Test all orientations
    let orientations: [UIImage.Orientation] = [
      .up, .down, .left, .right, .upMirrored, .downMirrored, .leftMirrored, .rightMirrored,
    ]

    for orientation in orientations {
      let testImage = UIImage(cgImage: originalImage.cgImage!, scale: 1.0, orientation: orientation)
      let normalized = normalizeImageOrientation(testImage)

      // All normalized images should have .up orientation
      XCTAssertEqual(
        normalized.imageOrientation, .up,
        "Image with \(orientation) orientation should normalize to .up")
      XCTAssertGreaterThan(normalized.size.width, 0, "Normalized image should have valid width")
      XCTAssertGreaterThan(normalized.size.height, 0, "Normalized image should have valid height")
    }
  }

  /// Tests that images already with .up orientation are returned unchanged.
  func testNormalizeImageOrientationAlreadyUp() {
    let size = CGSize(width: 100, height: 100)
    UIGraphicsBeginImageContext(size)
    UIColor.blue.setFill()
    UIRectFill(CGRect(origin: .zero, size: size))
    let originalImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()

    // Image should already have .up orientation
    XCTAssertEqual(originalImage.imageOrientation, .up)

    // Normalizing should return the same image instance (or equivalent)
    let normalized = normalizeImageOrientation(originalImage)
    XCTAssertEqual(normalized.imageOrientation, .up)
    XCTAssertEqual(normalized.size.width, originalImage.size.width)
    XCTAssertEqual(normalized.size.height, originalImage.size.height)
  }

  /// Tests that YOLO automatically normalizes orientation when processing UIImage.
  func testYOLOAutomaticOrientationNormalization() {
    // Create a test image with non-.up orientation
    let size = CGSize(width: 100, height: 100)
    UIGraphicsBeginImageContext(size)
    UIColor.green.setFill()
    UIRectFill(CGRect(origin: .zero, size: size))
    let baseImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()

    // Create UIImage with .left orientation (common from photo library)
    let rotatedImage = UIImage(cgImage: baseImage.cgImage!, scale: 1.0, orientation: .left)
    XCTAssertEqual(rotatedImage.imageOrientation, .left, "Test image should have .left orientation")

    // Create a mock YOLO instance to test the callAsFunction method
    // We'll verify that the internal normalization happens by checking the CIImage conversion
    // Since we can't easily test the full YOLO pipeline without a model, we test the normalization
    // function that YOLO uses internally
    let normalized = normalizeImageOrientation(rotatedImage)
    XCTAssertEqual(
      normalized.imageOrientation, .up, "YOLO should normalize orientation automatically")

    // Verify that converting to CIImage works correctly with normalized image
    let ciImage = CIImage(image: normalized)
    XCTAssertNotNil(ciImage, "Normalized image should convert to CIImage successfully")
    if let ciImage = ciImage {
      XCTAssertEqual(ciImage.extent.width, 100, accuracy: 0.1)
      XCTAssertEqual(ciImage.extent.height, 100, accuracy: 0.1)
    }
  }

  /// Tests that normalization preserves image scale.
  func testNormalizeImageOrientationPreservesScale() {
    let size = CGSize(width: 200, height: 200)
    UIGraphicsBeginImageContextWithOptions(size, false, 2.0)  // @2x scale
    UIColor.yellow.setFill()
    UIRectFill(CGRect(origin: .zero, size: size))
    let originalImage = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()

    let rotatedImage = UIImage(cgImage: originalImage.cgImage!, scale: 2.0, orientation: .right)
    let normalized = normalizeImageOrientation(rotatedImage)

    XCTAssertEqual(normalized.scale, 2.0, "Normalization should preserve image scale")
    XCTAssertEqual(normalized.imageOrientation, .up)
  }
}
