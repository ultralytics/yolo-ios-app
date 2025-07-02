// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import AVFoundation
import CoreVideo
import UIKit
@testable import YOLO

/// Tests for VideoCapture functionality
/// These tests focus on logic and interfaces rather than hardware-dependent features
class VideoCaptureTests: XCTestCase {
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        // Skip hardware-dependent tests in CI environment
        #if targetEnvironment(simulator)
            if ProcessInfo.processInfo.environment["CI"] != nil {
                throw XCTSkip("Skipping VideoCapture hardware tests in CI simulator environment")
            }
        #endif
    }
    
    // MARK: - Unit Tests (Logic Only)
    
    func testVideoCaptureInitialization() {
        let videoCapture = VideoCapture()
        
        // Test initial state
        XCTAssertNil(videoCapture.predictor)
        XCTAssertNil(videoCapture.previewLayer)
        XCTAssertNil(videoCapture.delegate)
        XCTAssertTrue(videoCapture.inferenceOK)
        XCTAssertEqual(videoCapture.longSide, 3)
        XCTAssertEqual(videoCapture.shortSide, 4)
        XCTAssertFalse(videoCapture.frameSizeCaptured)
    }
    
    func testAspectRatioProperties() {
        let videoCapture = VideoCapture()
        
        // Test setting aspect ratio
        videoCapture.longSide = 16
        videoCapture.shortSide = 9
        
        XCTAssertEqual(videoCapture.longSide, 16)
        XCTAssertEqual(videoCapture.shortSide, 9)
    }
    
    func testInferenceControl() {
        let videoCapture = VideoCapture()
        
        // Test inference control flag
        XCTAssertTrue(videoCapture.inferenceOK)
        
        videoCapture.inferenceOK = false
        XCTAssertFalse(videoCapture.inferenceOK)
        
        videoCapture.inferenceOK = true
        XCTAssertTrue(videoCapture.inferenceOK)
    }
    
    func testLastCapturedPhotoStorage() {
        let videoCapture = VideoCapture()
        
        // Test photo storage
        XCTAssertNil(videoCapture.lastCapturedPhoto)
        
        let testImage = UIImage(systemName: "camera")
        videoCapture.lastCapturedPhoto = testImage
        
        XCTAssertNotNil(videoCapture.lastCapturedPhoto)
        XCTAssertEqual(videoCapture.lastCapturedPhoto, testImage)
    }
    
    // MARK: - Integration Tests with Mocks
    
    func testVideoCaptureWithMockPredictor() async {
        let videoCapture = TestableVideoCapture()
        let mockPredictor = MockPredictor()
        let mockDelegate = await MockVideoCaptureDelegate()
        
        // Configure mock predictor
        mockPredictor.labels = ["person", "car", "bicycle"]
        mockPredictor.mockResult = YOLOResult(
            orig_shape: CGSize(width: 640, height: 480),
            boxes: [
                Box(
                    index: 0,
                    cls: "person",
                    conf: 0.9,
                    xywh: CGRect(x: 100, y: 100, width: 50, height: 100),
                    xywhn: CGRect(x: 0.156, y: 0.208, width: 0.078, height: 0.208)
                )
            ],
            speed: 15.5,
            names: mockPredictor.labels
        )
        
        // Set up video capture
        videoCapture.delegate = mockDelegate
        videoCapture.predictor = mockPredictor
        
        // Verify setup
        XCTAssertNotNil(videoCapture.predictor)
        XCTAssertNotNil(videoCapture.delegate)
        XCTAssertTrue(videoCapture.delegate === mockDelegate)
    }
    
    func testCameraSetupCompletion() {
        let videoCapture = TestableVideoCapture()
        let expectation = XCTestExpectation(description: "Camera setup completion")
        
        videoCapture.setUp(
            sessionPreset: .hd1280x720,
            position: .back,
            orientation: .portrait
        ) { success in
            XCTAssertTrue(success)
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Thread Safety Tests
    
    func testConcurrentAccess() {
        let videoCapture = VideoCapture()
        let expectation = XCTestExpectation(description: "Concurrent access")
        expectation.expectedFulfillmentCount = 10
        
        let group = DispatchGroup()
        
        for i in 0..<10 {
            group.enter()
            DispatchQueue.global().async {
                // Test concurrent property access
                _ = videoCapture.inferenceOK
                videoCapture.longSide = CGFloat(i)
                _ = videoCapture.lastCapturedPhoto
                
                group.leave()
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
}

// MARK: - Mock Delegate

@MainActor
class MockVideoCaptureDelegate: NSObject, VideoCaptureDelegate {
    var lastResult: YOLOResult?
    var lastInferenceTime: Double?
    var lastFPS: Double?
    var predictCallCount = 0
    var inferenceTimeCallCount = 0
    
    func onPredict(result: YOLOResult) {
        lastResult = result
        predictCallCount += 1
    }
    
    func onInferenceTime(speed: Double, fps: Double) {
        lastInferenceTime = speed
        lastFPS = fps
        inferenceTimeCallCount += 1
    }
}