// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreImage
import CoreML
import UIKit
import Vision
import XCTest
import AVFoundation

@testable import YOLO

/// Comprehensive test suite for validating all functions of the YOLO framework.
class YOLOTests: XCTestCase {
  // MARK: - Model Loading Tests
  
  /// Test that a valid detection model can be correctly loaded
  func testLoadValidDetectionModel() async throws {
    let expectation = XCTestExpectation(description: "Load detection model")
    
    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")
    
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
    
    await self.fulfillment(of: [expectation], timeout: 5.0)
  }

  /// Test that a segmentation model can be correctly loaded
  func testLoadSegmentationModel() async throws {
    let expectation = XCTestExpectation(description: "Load segmentation model")
    
    let modelURL = Bundle.module.url(forResource: "yolo11n-seg", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")
    
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
      
      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }

  // MARK: - YOLOResult Tests
  
  /// Test YOLOResult initialization and properties
  func testYOLOResultInitialization() {
    let result = YOLOResult(
      boxes: [YOLOBox(x: 10, y: 20, width: 100, height: 200, confidence: 0.9, classIndex: 0, name: "person")],
      masks: nil,
      probs: nil,
      keypoints: nil,
      obb: nil,
      speed: 15.0, 
      task: .detect,
      orig_shape: CGSize(width: 640, height: 640)
    )
    
    XCTAssertEqual(result.boxes?.count, 1)
    XCTAssertEqual(result.boxes?[0].name, "person")
    XCTAssertEqual(result.boxes?[0].confidence, 0.9)
    XCTAssertEqual(result.speed, 15.0)
    XCTAssertEqual(result.task, .detect)
    XCTAssertEqual(result.orig_shape.width, 640)
    XCTAssertEqual(result.orig_shape.height, 640)
    XCTAssertNil(result.masks)
    XCTAssertNil(result.probs)
    XCTAssertNil(result.keypoints)
    XCTAssertNil(result.obb)
  }
  
  // MARK: - YOLOBox Tests
  
  /// Test YOLOBox properties and calculations
  func testYOLOBoxProperties() {
    let box = YOLOBox(x: 100, y: 150, width: 200, height: 300, confidence: 0.85, classIndex: 2, name: "car")
    
    XCTAssertEqual(box.x, 100)
    XCTAssertEqual(box.y, 150)
    XCTAssertEqual(box.width, 200)
    XCTAssertEqual(box.height, 300)
    XCTAssertEqual(box.confidence, 0.85)
    XCTAssertEqual(box.classIndex, 2)
    XCTAssertEqual(box.name, "car")
    
    // Test computed properties
    XCTAssertEqual(box.rect, CGRect(x: 100, y: 150, width: 200, height: 300))
    XCTAssertEqual(box.centerX, 200)
    XCTAssertEqual(box.centerY, 300)
  }
  
  // MARK: - Task Type Tests
  
  /// Test YOLO task string representations
  func testYOLOTaskStringRepresentation() {
    XCTAssertEqual(YOLOTask.detect.stringValue, "detect")
    XCTAssertEqual(YOLOTask.segment.stringValue, "segment")
    XCTAssertEqual(YOLOTask.classify.stringValue, "classify")
    XCTAssertEqual(YOLOTask.pose.stringValue, "pose")
    XCTAssertEqual(YOLOTask.obb.stringValue, "obb")
  }

  // MARK: - Processing Pipeline Tests

  /// Helper method to get a test image
  @MainActor
  private func getTestImage(size: CGSize = CGSize(width: 640, height: 640)) -> UIImage? {
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

    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          // Process the image after model is loaded
          guard let yolo = yolo else { return }

          Task {
            let yoloResult = yolo(ciImage)

            // Validate results
            XCTAssertNotNil(yoloResult)
            XCTAssertEqual(yoloResult.orig_shape.width, ciImage.extent.width)
            XCTAssertEqual(yoloResult.orig_shape.height, ciImage.extent.height)
            XCTAssertNotNil(yoloResult.boxes)
            XCTAssertGreaterThan(yoloResult.speed, 0)

            expectation.fulfill()
          }

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

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

    let modelURL = Bundle.module.url(forResource: "yolo11n-cls", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .classify) { result in
        switch result {
        case .success(_):
          guard let yolo = yolo else { return }

          Task {
            let yoloResult = yolo(ciImage)

            // Validate classification results
            XCTAssertNotNil(yoloResult)
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

      await fulfillment(of: [expectation], timeout: 10.0)
    }
  }
  
  /// Test error handling with invalid image input
  @MainActor
  func testErrorHandlingWithInvalidImageInput() async throws {
    let expectation = XCTestExpectation(description: "Test invalid image input")
    
    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")
    
    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          guard let yolo = yolo else { return }
          
          Task {
            // Create an invalid CIImage (empty)
            let invalidImage = CIImage()
            
            // Process invalid image - should handle gracefully
            let result = yolo(invalidImage)
            
            // Should return empty result but not crash
            XCTAssertNotNil(result)
            XCTAssertTrue(result.boxes?.isEmpty ?? true)
            
            expectation.fulfill()
          }
          
        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }
      
      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }

  // MARK: - Configuration Tests

  /// Test that confidence threshold setting functions correctly
  func testConfidenceThresholdSetting() async throws {
    let expectation = XCTestExpectation(description: "Confidence threshold setting")

    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")

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

          // Verify new threshold
          XCTAssertEqual(predictor.confidenceThreshold, newThreshold)

          expectation.fulfill()

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }

  /// Test that IoU threshold setting functions correctly
  func testIoUThresholdSetting() async throws {
    let expectation = XCTestExpectation(description: "IoU threshold setting")

    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")

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

          // Verify new threshold
          XCTAssertEqual(predictor.iouThreshold, newThreshold)

          expectation.fulfill()

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }
  
  /// Test configuration persistence after inference
  @MainActor
  func testConfigurationPersistenceAfterInference() async throws {
    let expectation = XCTestExpectation(description: "Test configuration persistence")
    
    guard let testImage = getTestImage(),
          let ciImage = CIImage(image: testImage) else {
      XCTFail("Failed to create test image")
      return
    }
    
    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")
    
    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          guard let yolo = yolo, let predictor = yolo.predictor as? BasePredictor else { return }
          
          // Set custom configuration
          let customConfidence = 0.7
          let customIOU = 0.6
          predictor.setConfidenceThreshold(confidence: customConfidence)
          predictor.setIouThreshold(iou: customIOU)
          
          Task {
            // Run inference
            let _ = yolo(ciImage)
            
            // Check if settings are maintained
            XCTAssertEqual(predictor.confidenceThreshold, customConfidence)
            XCTAssertEqual(predictor.iouThreshold, customIOU)
            
            expectation.fulfill()
          }
          
        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }
      
      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }

  // MARK: - YOLOCamera Tests

  /// Test that YOLOCamera component can be initialized
  @MainActor
  func testYOLOCameraInitialization() async throws {
    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")

    if let url = modelURL {
      let yoloCamera = YOLOCamera(modelPathOrName: url.path, task: .detect, cameraPosition: .back)

      // Validate component properties
      XCTAssertEqual(yoloCamera.task, .detect)
      XCTAssertEqual(yoloCamera.cameraPosition, .back)
      XCTAssertNotNil(yoloCamera.body)
    }
  }
  
  /// Test YOLOCamera with different parameter combinations
  @MainActor
  func testYOLOCameraWithDifferentParameters() async throws {
    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")
    
    if let url = modelURL {
      // Test with front camera
      let frontCamera = YOLOCamera(modelPathOrName: url.path, task: .detect, cameraPosition: .front)
      XCTAssertEqual(frontCamera.cameraPosition, .front)
      
      // Test with different task
      let segCamera = YOLOCamera(modelPathOrName: url.path, task: .detect, cameraPosition: .back)
      XCTAssertEqual(segCamera.task, .detect)
    }
  }

  // MARK: - YOLOView Tests

  /// Test that YOLOView component can be initialized
  @MainActor
  func testYOLOViewInitialization() async throws {
    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")

    if let url = modelURL {
      // Initialize YOLOView
      let frame = CGRect(x: 0, y: 0, width: 640, height: 480)
      let yoloView = YOLOView(frame: frame, modelPathOrName: url.path, task: .detect)

      // Validate component properties
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

    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")

    if let url = modelURL {
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success(_):
          guard let yolo = yolo else { return }

          // Measure performance
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
            // Check if inference time is reasonable
            XCTAssertLessThan(averageTime, 1.0, "Inference is too slow (> 1 second)")
            expectation.fulfill()
          }

        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          expectation.fulfill()
        }
      }

      await self.fulfillment(of: [expectation], timeout: 30.0)
    }
  }
  
  /// Test concurrent model inference
  @MainActor
  func testConcurrentModelInference() async throws {
    guard let testImage = getTestImage(),
          let ciImage = CIImage(image: testImage) else {
      XCTFail("Failed to create test image")
      return
    }
    
    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")
    
    if let url = modelURL {
      let initExpectation = XCTestExpectation(description: "Initialize model")
      
      var yolo: YOLO? = nil
      yolo = YOLO(url.path, task: .detect) { result in
        switch result {
        case .success():
          initExpectation.fulfill()
        case .failure(let error):
          XCTFail("Failed to load model: \(error)")
          initExpectation.fulfill()
        }
      }
      
      await fulfillment(of: [initExpectation], timeout: 5.0)
      
      guard let yolo = yolo else {
        XCTFail("YOLO instance is nil")
        return
      }
      
      // Run multiple inferences concurrently
      let concurrentInferenceCount = 5
      let inferenceExpectations = (0..<concurrentInferenceCount).map { 
        XCTestExpectation(description: "Concurrent inference \($0)")
      }
      
      await withTaskGroup(of: Void.self) { group in
        for i in 0..<concurrentInferenceCount {
          group.addTask {
            let result = yolo(ciImage)
            XCTAssertNotNil(result)
            inferenceExpectations[i].fulfill()
          }
        }
      }
      
      await fulfillment(of: inferenceExpectations, timeout: 10.0)
    }
  }

  // MARK: - Error Handling Tests

  /// Test error handling when task type and model do not match
  func testTaskMismatchError() async throws {
    let expectation = XCTestExpectation(description: "Task mismatch error")

    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")

    if let url = modelURL {
      // Use detection model with classification task - should fail
      let _ = YOLO(url.path, task: .classify) { result in
        switch result {
        case .success(_):
          XCTFail("Should not succeed with mismatched task type")
        case .failure(let error):
          XCTAssertNotNil(error)
          expectation.fulfill()
        }
      }

      await fulfillment(of: [expectation], timeout: 5.0)
    }
  }
  
  /// Test multiple tasks with the same model
  @MainActor
  func testMultipleTasksWithSameModel() async throws {
    // Get model path
    let modelURL = Bundle.module.url(forResource: "yolo11n", withExtension: "mlpackage", subdirectory: "Resources")
    XCTAssertNotNil(modelURL, "Test model file not found")
    
    guard let url = modelURL else { return }
    
    // Create test image
    guard let testImage = getTestImage(), 
          let ciImage = CIImage(image: testImage) else {
      XCTFail("Failed to create test image")
      return
    }
    
    // Test with detection task
    let expectation1 = XCTestExpectation(description: "Load model for detection task")
    var yoloDetect: YOLO? = nil
    
    yoloDetect = YOLO(url.path, task: .detect) { result in
      switch result {
      case .success():
        // Process image
        guard let yolo = yoloDetect else { return }
        
        Task {
          let result = yolo(ciImage)
          XCTAssertNotNil(result)
          XCTAssertEqual(result.task, .detect)
          expectation1.fulfill()
        }
        
      case .failure(let error):
        XCTFail("Failed to load model for detection: \(error)")
        expectation1.fulfill()
      }
    }
    
    await fulfillment(of: [expectation1], timeout: 10.0)
    
    // Test with the same model but different task (should fail)
    let expectation2 = XCTestExpectation(description: "Load model with incorrect task")
    
    let _ = YOLO(url.path, task: .classify) { result in
      switch result {
      case .success():
        XCTFail("Should not succeed with incorrect task")
        expectation2.fulfill()
      case .failure(let error):
        XCTAssertNotNil(error)
        expectation2.fulfill()
      }
    }
    
    await fulfillment(of: [expectation2], timeout: 5.0)
  }
}
