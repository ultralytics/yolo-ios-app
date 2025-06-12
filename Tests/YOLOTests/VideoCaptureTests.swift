// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import AVFoundation
import Vision
@testable import YOLO

/// Minimal tests for VideoCapture functionality
class VideoCaptureTests: XCTestCase {
    
    func testVideoCaptureInitialization() {
        """Test VideoCapture initialization"""
        let videoCapture = VideoCapture()
        
        XCTAssertNil(videoCapture.predictor)
        XCTAssertNil(videoCapture.previewLayer)
        XCTAssertNil(videoCapture.delegate)
        XCTAssertNil(videoCapture.captureDevice)
        XCTAssertNotNil(videoCapture.captureSession)
        XCTAssertNil(videoCapture.videoInput)
        XCTAssertNotNil(videoCapture.videoOutput)
        XCTAssertNotNil(videoCapture.photoOutput)
        XCTAssertNotNil(videoCapture.cameraQueue)
        XCTAssertNil(videoCapture.lastCapturedPhoto)
        XCTAssertTrue(videoCapture.inferenceOK)
        XCTAssertEqual(videoCapture.longSide, 3)
        XCTAssertEqual(videoCapture.shortSide, 4)
        XCTAssertFalse(videoCapture.frameSizeCaptured)
    }
    
    func testVideoCaptureSessionConfiguration() {
        """Test capture session properties"""
        let videoCapture = VideoCapture()
        
        XCTAssertNotNil(videoCapture.captureSession)
        XCTAssertFalse(videoCapture.captureSession.isRunning)
    }
    
    func testVideoCaptureOutputConfiguration() {
        """Test video output configuration"""
        let videoCapture = VideoCapture()
        
        XCTAssertNotNil(videoCapture.videoOutput)
        XCTAssertTrue(videoCapture.videoOutput.alwaysDiscardsLateVideoFrames)
        
        XCTAssertNotNil(videoCapture.photoOutput)
        XCTAssertTrue(videoCapture.photoOutput.isHighResolutionCaptureEnabled)
    }
    
    func testVideoCaptureStartStop() {
        """Test start and stop methods don't crash"""
        let videoCapture = VideoCapture()
        
        // These should not crash even without proper setup
        videoCapture.start()
        videoCapture.stop()
        
        XCTAssertTrue(true) // Test passes if no crash
    }
    
    func testVideoCaptureZoomRatio() {
        """Test setZoomRatio method handles nil captureDevice gracefully"""
        let videoCapture = VideoCapture()
        
        // Should not crash with nil captureDevice
        videoCapture.setZoomRatio(ratio: 2.0)
        
        XCTAssertTrue(true) // Test passes if no crash
    }
    
    func testVideoCaptureInferenceFlag() {
        """Test inferenceOK flag can be set and read"""
        let videoCapture = VideoCapture()
        
        XCTAssertTrue(videoCapture.inferenceOK)
        
        videoCapture.inferenceOK = false
        XCTAssertFalse(videoCapture.inferenceOK)
        
        videoCapture.inferenceOK = true
        XCTAssertTrue(videoCapture.inferenceOK)
    }
    
    func testVideoCaptureDimensionProperties() {
        """Test frame dimension properties"""
        let videoCapture = VideoCapture()
        
        XCTAssertEqual(videoCapture.longSide, 3)
        XCTAssertEqual(videoCapture.shortSide, 4)
        XCTAssertFalse(videoCapture.frameSizeCaptured)
        
        videoCapture.longSide = 1920
        videoCapture.shortSide = 1080
        videoCapture.frameSizeCaptured = true
        
        XCTAssertEqual(videoCapture.longSide, 1920)
        XCTAssertEqual(videoCapture.shortSide, 1080)
        XCTAssertTrue(videoCapture.frameSizeCaptured)
    }
    
    func testVideoCaptureUpdateVideoOrientation() {
        """Test updateVideoOrientation method handles empty connections gracefully"""
        let videoCapture = VideoCapture()
        
        // Should not crash even without proper connection setup
        videoCapture.updateVideoOrientation(orientation: .portrait)
        videoCapture.updateVideoOrientation(orientation: .landscapeLeft)
        videoCapture.updateVideoOrientation(orientation: .landscapeRight)
        videoCapture.updateVideoOrientation(orientation: .portraitUpsideDown)
        
        XCTAssertTrue(true) // Test passes if no crash
    }
    
    func testVideoCaptureAsResultsListener() {
        """Test VideoCapture conforms to ResultsListener"""
        let videoCapture = VideoCapture()
        
        XCTAssertTrue(videoCapture is ResultsListener)
        
        // Test on(result:) method doesn't crash
        let result = YOLOResult(orig_shape: CGSize(width: 640, height: 480), boxes: [], speed: 0.1, names: [])
        videoCapture.on(result: result)
        
        XCTAssertTrue(true) // Test passes if no crash
    }
    
    func testVideoCaptureAsInferenceTimeListener() {
        """Test VideoCapture conforms to InferenceTimeListener"""
        let videoCapture = VideoCapture()
        
        XCTAssertTrue(videoCapture is InferenceTimeListener)
        
        // Test on(inferenceTime:fpsRate:) method doesn't crash
        videoCapture.on(inferenceTime: 25.5, fpsRate: 30.0)
        
        XCTAssertTrue(true) // Test passes if no crash
    }
    
    func testVideoCaptureQueueCreation() {
        """Test camera queue is properly created"""
        let videoCapture = VideoCapture()
        
        XCTAssertNotNil(videoCapture.cameraQueue)
        
        // Test queue can execute tasks
        let expectation = self.expectation(description: "Camera queue task")
        videoCapture.cameraQueue.async {
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 1.0)
    }
    
    func testVideoCaptureWeakDelegate() {
        """Test delegate is weak reference"""
        let videoCapture = VideoCapture()
        
        class MockDelegate: VideoCaptureDelegate {
            func onPredict(result: YOLOResult) {}
            func onInferenceTime(speed: Double, fps: Double) {}
        }
        
        var delegate: MockDelegate? = MockDelegate()
        videoCapture.delegate = delegate
        
        XCTAssertNotNil(videoCapture.delegate)
        
        delegate = nil
        // Delegate should become nil because it's weak
        XCTAssertNil(videoCapture.delegate)
    }
}

// MARK: - Tests for VideoCapture utility functions

class VideoCaptureUtilityTests: XCTestCase {
    
    func testBestCaptureDeviceFunction() {
        """Test bestCaptureDevice function handles camera selection"""
        // This test might fail on simulator where cameras aren't available
        // but should not crash
        
        // Test with back camera position
        let backDevice = bestCaptureDevice(position: .back)
        XCTAssertNotNil(backDevice)
        XCTAssertEqual(backDevice.position, .back)
        
        // Test with front camera position if available
        if AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) != nil {
            let frontDevice = bestCaptureDevice(position: .front)
            XCTAssertNotNil(frontDevice)
            XCTAssertEqual(frontDevice.position, .front)
        }
    }
}

// MARK: - Mock Delegate for Testing

@MainActor
class MockVideoCaptureDelegate: VideoCaptureDelegate {
    var lastResult: YOLOResult?
    var lastSpeed: Double?
    var lastFPS: Double?
    
    func onPredict(result: YOLOResult) {
        lastResult = result
    }
    
    func onInferenceTime(speed: Double, fps: Double) {
        lastSpeed = speed
        lastFPS = fps
    }
}
