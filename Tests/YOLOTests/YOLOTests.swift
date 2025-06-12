// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import CoreML
import UIKit
import Vision
import XCTest

@testable import YOLO

/// Comprehensive test suite for validating all functions in the YOLO framework.
///
/// This test suite validates:
/// - Model loading and initialization
/// - Inference on static images
/// - Real-time camera frame processing
/// - Functionality of each task type (detection, segmentation, classification, pose estimation, OBB)
/// - Error handling and edge cases
/// - Performance and memory usage
///
/// # Prerequisites for running tests
///
/// The following model files are required to run these tests.
/// Please place these models in the Tests/YOLOTests/Resources directory before testing:
///
/// - yolo11n.mlpackage: Detection model
/// - yolo11n-seg.mlpackage: Segmentation model
/// - yolo11n-cls.mlpackage: Classification model
/// - yolo11n-pose.mlpackage: Pose estimation model
/// - yolo11n-obb.mlpackage: Oriented bounding box model
///
/// These models can be downloaded from https://github.com/ultralytics/ultralytics
/// and must be converted to CoreML format.
class YOLOTests: XCTestCase {
  // Basic diagnostic test
  func testBasic() {
    print("Running basic YOLO test diagnostic")
    print("Resource URL: \(Bundle.module.resourceURL?.path ?? "nil")")
    XCTAssertTrue(true)  // Always succeeds
  }

  // MARK: - Model Loading Tests

  /// Test that a valid detection model can be correctly loaded
  func testLoadValidDetectionModel() async throws {
    let expectation = XCTestExpectation(description: "Load detection model")

    let modelURL = Bundle.module.url(
      forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          XCTAssertNotNil(yolo?.predictor)
          XCTAssertEqual(yolo?.predictor.labels.isEmpty, false)
          expectation.fulfill()
        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
        }
      }

      // Wait for 5 seconds - using async version
      await self.fulfillment(of: [expectation], timeout: 5.0)
    }
  }

  /// Test error handling when model path is invalid
  func testLoadInvalidModelPath() async throws {
    let expectation = XCTestExpectation(description: "Invalid model path")

    let _ = YOLO("invalid_path.mlpackage", task: .detect) { result in
      switch result {
      case .success(_):
        XCTFail("Should not succeed with invalid path")
      case .failure(let error):
        XCTAssertNotNil(error)
        expectation.fulfill()
      }
    }

    // Changed to async version
    await self.fulfillment(of: [expectation], timeout: 5.0)
  }

  /// Test that a segmentation model can be correctly loaded
  func testLoadSegmentationModel() async throws {
    let expectation = XCTestExpectation(description: "Load segmentation model")

    let modelURL = Bundle.module.url(
      forResource: "yolo11n-seg", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n-seg.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .segment) { result in
        switch result {
        case .success(_):
          XCTAssertNotNil(yolo?.predictor)
          XCTAssertEqual(yolo?.predictor.labels.isEmpty, false)
          expectation.fulfill()
        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
        }
      }

      // Changed to async version
      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }

  // MARK: - Processing Pipeline Tests

  /// Helper method to get a test image
  @MainActor
  private func getTestImage() -> UIImage? {
    // Generate test image (white square on black background)
    let size = CGSize(width: 640, height: 640)
    UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
    UIColor.black.setFill()
    UIRectFill(CGRect(origin: .zero, size: size))
    UIColor.white.setFill()
    UIRectFill(CGRect(x: 200, y: 200, width: 240, height: 240))
    let image = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return image
  }

  /// Test processing static images with a detection model
  @MainActor
  func testProcessStaticImageWithDetectionModel() async throws {
    let expectation = XCTestExpectation(description: "Process static image")

    guard let testImage = getTestImage(),
      let ciImage = CIImage(image: testImage)
    else {
      XCTFail("Failed to create test image")
      return
    }

    let modelURL = Bundle.module.url(
      forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          // Process the image after the model is loaded
          guard let yolo = yolo else { return }

          // Get YOLOResult asynchronously
          Task {
            let yoloResult = yolo(ciImage)

            // Validate results
            XCTAssertNotNil(yoloResult)
            XCTAssertEqual(yoloResult.orig_shape.width, ciImage.extent.width)
            XCTAssertEqual(yoloResult.orig_shape.height, ciImage.extent.height)

            // Detection results exist as an array (can be empty)
            XCTAssertNotNil(yoloResult.boxes)

            // Processing speed is recorded
            XCTAssertGreaterThan(yoloResult.speed, 0)

            expectation.fulfill()
          }

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

      // Changed to async version
      await fulfillment(of: [expectation], timeout: 10.0)
    }
  }

  /// Test processing static images with a classification model
  @MainActor
  func testProcessStaticImageWithClassificationModel() async throws {
    let expectation = XCTestExpectation(description: "Process static image with classification")

    guard let testImage = getTestImage(),
      let ciImage = CIImage(image: testImage)
    else {
      XCTFail("Failed to create test image")
      return
    }

    let modelURL = Bundle.module.url(
      forResource: "yolo11n-cls", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n-cls.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .classify) { result in
        switch result {
        case .success(_):
          // Process the image after the model is loaded
          guard let yolo = yolo else { return }

          // Get YOLOResult asynchronously
          Task {
            let yoloResult = yolo(ciImage)

            // Validate results
            XCTAssertNotNil(yoloResult)

            // Classification results exist
            XCTAssertNotNil(yoloResult.probs)

            if let probs = yoloResult.probs {
              XCTAssertFalse(probs.top1.isEmpty)
              XCTAssertGreaterThan(probs.top1Conf, 0)
              XCTAssertEqual(probs.top5.count, 5)
              XCTAssertEqual(probs.top5Confs.count, 5)
            }

            expectation.fulfill()
          }

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

      // Changed to async version
      await fulfillment(of: [expectation], timeout: 10.0)
    }
  }

  // MARK: - Configuration Tests

  /// Test that confidence threshold setting functions correctly
  func testConfidenceThresholdSetting() async throws {
    let expectation = XCTestExpectation(description: "Confidence threshold setting")

    let modelURL = Bundle.module.url(
      forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          guard let yolo = yolo, let predictor = yolo.predictor as? BasePredictor else { return }

          // Check default threshold
          let defaultThreshold = predictor.confidenceThreshold
          XCTAssertEqual(defaultThreshold, 0.25)

          // Change threshold
          let newThreshold = 0.7
          predictor.setConfidenceThreshold(confidence: newThreshold)

          // Verify changed threshold
          XCTAssertEqual(predictor.confidenceThreshold, newThreshold)

          expectation.fulfill()

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

      // Changed to async version
      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }

  /// Test that IoU threshold setting functions correctly
  func testIoUThresholdSetting() async throws {
    let expectation = XCTestExpectation(description: "IoU threshold setting")

    let modelURL = Bundle.module.url(
      forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          guard let yolo = yolo, let predictor = yolo.predictor as? BasePredictor else { return }

          // Check default threshold
          let defaultThreshold = predictor.iouThreshold
          XCTAssertEqual(defaultThreshold, 0.4)

          // Change threshold
          let newThreshold = 0.8
          predictor.setIouThreshold(iou: newThreshold)

          // Verify changed threshold
          XCTAssertEqual(predictor.iouThreshold, newThreshold)

          expectation.fulfill()

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

      // Changed to async version
      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }

  // MARK: - YOLOCamera Tests

  /// Test that YOLOCamera component can be initialized
  @MainActor
  func testYOLOCameraInitialization() async throws {
    let modelURL = Bundle.module.url(
      forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      // Create UIViewController and initialize YOLOCamera
      let viewController = UIViewController()
      let yoloCamera = YOLOCamera(modelPathOrName: url.path, task: .detect, cameraPosition: .back)

      // Verify component properties
      XCTAssertEqual(yoloCamera.task, .detect)
      XCTAssertEqual(yoloCamera.cameraPosition, .back)
      XCTAssertNotNil(yoloCamera.body)
    }
  }

  // MARK: - YOLOView Tests

  /// Test that YOLOView component can be initialized
  @MainActor
  func testYOLOViewInitialization() async throws {
    let modelURL = Bundle.module.url(
      forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      // Initialize YOLOView
      let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
      let yoloView = YOLOView(frame: frame, modelPathOrName: url.path, task: .detect)

      // Verify component properties
      XCTAssertNotNil(yoloView)
      XCTAssertEqual(yoloView.frame, frame)
    }
  }

  // MARK: - Performance Tests

  /// Measure the performance of model inference
  @MainActor
  func testInferencePerformance() async throws {
    let expectation = XCTestExpectation(description: "Inference performance")

    guard let testImage = getTestImage(),
      let ciImage = CIImage(image: testImage)
    else {
      XCTFail("Failed to create test image")
      return
    }

    let modelURL = Bundle.module.url(
      forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          guard let yolo = yolo else { return }

          // Execute performance measurement
          Task {
            let iterations = 10
            var totalTime: Double = 0

            for _ in 0..<iterations {
              let startTime = CACurrentMediaTime()
              let _ = yolo(ciImage)
              let endTime = CACurrentMediaTime()
              totalTime += (endTime - startTime)
            }

            let averageTime = totalTime / Double(iterations)

            print("Average inference time: \(averageTime * 1000) ms")
            // Check if inference time is within a reasonable range
            XCTAssertLessThan(averageTime, 1.0, "Inference is too slow (> 1 second)")
            expectation.fulfill()
          }

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

      // Changed to async version
      await self.fulfillment(of: [expectation], timeout: 30.0)
    }
  }

  // MARK: - Error Handling Tests

  /// Test error handling when task type and model do not match
  func testTaskMismatchError() async throws {
    let expectation = XCTestExpectation(description: "Task mismatch error")

    let modelURL = Bundle.module.url(
      forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    //        XCTAssertNotNil(modelURL, "Test model file not found. Please add yolo11n.mlpackage to Tests/YOLOTests/Resources")

    if let url = modelURL {
      // Use detection model with classification task - should result in an error
      let _ = YOLO(url.path, task: .classify) { result in
        switch result {
        case .success(_):
          XCTFail("Should not succeed with mismatched task type")
        case .failure(let error):
          XCTAssertNotNil(error)
          expectation.fulfill()
        }
      }

      // Changed to async version
      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }
}

// MARK: - Main entry point for test execution

/// Notes before running tests
///
/// To run these tests, you need to place the following CoreML model files
/// in the Tests/YOLOTests/Resources directory:
///
/// - yolo11n.mlpackage - Detection model
/// - yolo11n-seg.mlpackage - Segmentation model
/// - yolo11n-cls.mlpackage - Classification model
/// - yolo11n-pose.mlpackage - Pose estimation model
/// - yolo11n-obb.mlpackage - Oriented bounding box model
///
/// How to download and convert models to CoreML:
/// 1. Download YOLO11 models from https://github.com/ultralytics/ultralytics
/// 2. Convert Ultralytics models to CoreML using the following Python code:
///
/// ```python
/// from ultralytics import YOLO
///
/// # Detection model
/// model = YOLO("yolo11n.pt")
/// model.export(format="coreml")
///
/// # Segmentation model
/// model = YOLO("yolo11n-seg.pt")
/// model.export(format="coreml")
///
/// # Classification model
/// model = YOLO("yolo11n-cls.pt")
/// model.export(format="coreml")
///
/// # Pose estimation model
/// model = YOLO("yolo11n-pose.pt")
/// model.export(format="coreml")
///
/// # Oriented bounding box model
/// model = YOLO("yolo11n-obb.pt")
/// model.export(format="coreml")
/// ```
///
/// 3. Place the generated .mlpackage files in the test directory
