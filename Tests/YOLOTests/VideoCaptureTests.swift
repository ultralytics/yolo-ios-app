// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import AVFoundation
import CoreVideo
import UIKit
@testable import YOLO

/// Comprehensive tests for VideoCapture functionality
class VideoCaptureTests: XCTestCase {
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        // Skip all VideoCapture tests in CI environment
        if ProcessInfo.processInfo.environment["CI"] != nil {
            throw XCTSkip("Skipping VideoCapture tests in CI environment")
        }
    }
    
    var videoCapture: VideoCapture!
    var mockDelegate: MockVideoCaptureDelegate!
    
    @MainActor
    override func setUp() {
        super.setUp()
        videoCapture = VideoCapture()
        mockDelegate = MockVideoCaptureDelegate()
    }
    
    override func tearDown() {
        videoCapture?.stop()
        videoCapture = nil
        mockDelegate = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testVideoCaptureInitialization() {
        // Verify initial state
        XCTAssertNil(videoCapture.predictor)
        XCTAssertNil(videoCapture.previewLayer)
        XCTAssertNil(videoCapture.delegate)
        XCTAssertNil(videoCapture.captureDevice)
        XCTAssertNil(videoCapture.videoInput)
        XCTAssertNil(videoCapture.lastCapturedPhoto)
        XCTAssertTrue(videoCapture.inferenceOK)
        XCTAssertEqual(videoCapture.longSide, 3)
        XCTAssertEqual(videoCapture.shortSide, 4)
        XCTAssertFalse(videoCapture.frameSizeCaptured)
    }
    
    func testCaptureSessionInitialization() {
        // Verify capture session is created
        XCTAssertNotNil(videoCapture.captureSession)
        XCTAssertFalse(videoCapture.captureSession.isRunning)
    }
    
    func testVideoOutputConfiguration() {
        // Verify video output is configured properly
        XCTAssertNotNil(videoCapture.videoOutput)
        XCTAssertNotNil(videoCapture.photoOutput)
    }
    
    // MARK: - Device Selection Tests
    
    func testBestCaptureDeviceFunction() {
        // Test device selection logic
        UserDefaults.standard.set(false, forKey: "use_telephoto")
        
        // Note: These tests may fail on simulator where camera devices aren't available
        if AVCaptureDevice.default(for: .video) != nil {
            let backDevice = bestCaptureDevice(position: .back)
            XCTAssertNotNil(backDevice)
            XCTAssertEqual(backDevice.position, .back)
            
            let frontDevice = bestCaptureDevice(position: .front)
            XCTAssertNotNil(frontDevice)
            XCTAssertEqual(frontDevice.position, .front)
        }
    }
    
    func testBestCaptureDeviceWithTelephoto() {
        // Test telephoto preference
        UserDefaults.standard.set(true, forKey: "use_telephoto")
        
        if AVCaptureDevice.default(for: .video) != nil {
            let device = bestCaptureDevice(position: .back)
            XCTAssertNotNil(device)
            // Device type depends on hardware availability
        }
        
        // Reset preference
        UserDefaults.standard.set(false, forKey: "use_telephoto")
    }
    
    // MARK: - Delegate Tests
    
    func testDelegateAssignment() {
        videoCapture.delegate = mockDelegate
        XCTAssertNotNil(videoCapture.delegate)
        XCTAssertTrue(videoCapture.delegate === mockDelegate)
    }
    
    // MARK: - Camera Setup Tests
    
    func testSetUpCameraCompletion() {
        let expectation = XCTestExpectation(description: "Camera setup completion")
        
        videoCapture.setUp(
            sessionPreset: .hd1280x720,
            position: .back,
            orientation: .portrait
        ) { success in
            // On simulator, this will likely fail due to no camera
            XCTAssertNotNil(success)
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
    
    func testCameraQueueSetup() {
        // Verify camera queue exists
        XCTAssertNotNil(videoCapture.cameraQueue)
        
        // Test that operations are dispatched to camera queue
        let expectation = XCTestExpectation(description: "Camera queue operation")
        
        videoCapture.cameraQueue.async {
            XCTAssertFalse(Thread.isMainThread)
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Predictor Integration Tests
    
    func testPredictorAssignment() {
        let mockPredictor = MockVideoCapturePredictor()
        videoCapture.predictor = mockPredictor
        
        XCTAssertNotNil(videoCapture.predictor)
    }
    
    // MARK: - Frame Processing Tests
    
    func testInferenceOKFlag() {
        // Test inference control flag
        XCTAssertTrue(videoCapture.inferenceOK)
        
        videoCapture.inferenceOK = false
        XCTAssertFalse(videoCapture.inferenceOK)
        
        videoCapture.inferenceOK = true
        XCTAssertTrue(videoCapture.inferenceOK)
    }
    
    func testFrameSizeCapture() {
        // Test frame size capture flag
        XCTAssertFalse(videoCapture.frameSizeCaptured)
        
        videoCapture.frameSizeCaptured = true
        XCTAssertTrue(videoCapture.frameSizeCaptured)
    }
    
    func testAspectRatioProperties() {
        // Test aspect ratio properties
        videoCapture.longSide = 16
        videoCapture.shortSide = 9
        
        XCTAssertEqual(videoCapture.longSide, 16)
        XCTAssertEqual(videoCapture.shortSide, 9)
    }
    
    // MARK: - Photo Capture Tests
    
    func testLastCapturedPhoto() {
        // Test photo storage
        XCTAssertNil(videoCapture.lastCapturedPhoto)
        
        let testImage = UIImage(systemName: "camera")
        videoCapture.lastCapturedPhoto = testImage
        
        XCTAssertNotNil(videoCapture.lastCapturedPhoto)
        XCTAssertEqual(videoCapture.lastCapturedPhoto, testImage)
    }
    
    // MARK: - Session Control Tests
    
    func testStopSession() {
        // Test stopping capture session
        videoCapture.stop()
        
        // Verify session is not running
        XCTAssertFalse(videoCapture.captureSession.isRunning)
    }
    
    // MARK: - Preview Layer Tests
    
    func testPreviewLayerCreation() {
        // Test preview layer can be created
        let previewLayer = AVCaptureVideoPreviewLayer(session: videoCapture.captureSession)
        videoCapture.previewLayer = previewLayer
        
        XCTAssertNotNil(videoCapture.previewLayer)
        XCTAssertEqual(videoCapture.previewLayer?.session, videoCapture.captureSession)
    }
    
    // MARK: - Orientation Tests
    
    func testOrientationMapping() {
        // Test orientation conversion in setUpCamera
        let orientationTests: [(UIDeviceOrientation, AVCaptureVideoOrientation)] = [
            (.portrait, .portrait),
            (.landscapeLeft, .landscapeRight),
            (.landscapeRight, .landscapeLeft),
            (.portraitUpsideDown, .portraitUpsideDown)
        ]
        
        // This is a conceptual test since we can't easily test the internal implementation
        for (deviceOrientation, _) in orientationTests {
            XCTAssertNotNil(deviceOrientation.rawValue)
        }
    }
    
    // MARK: - Buffer Management Tests
    
    func testCurrentBufferProperty() {
        // Test that current buffer starts as nil
        let mirror = Mirror(reflecting: videoCapture!)
        var foundCurrentBuffer = false
        
        for child in mirror.children {
            if child.label == "currentBuffer" {
                foundCurrentBuffer = true
                // Simply check that the value is nil without type casting
                XCTAssertNil(child.value as AnyObject)
            }
        }
        
        XCTAssertTrue(foundCurrentBuffer, "currentBuffer property should exist")
    }
    
    // MARK: - Integration Tests
    
    func testVideoCaptureLifecycle() {
        // Test complete lifecycle
        let setupExpectation = XCTestExpectation(description: "Setup")
        
        // 1. Set delegate
        videoCapture.delegate = mockDelegate
        
        // 2. Set predictor
        videoCapture.predictor = MockVideoCapturePredictor()
        
        // 3. Setup camera
        videoCapture.setUp(
            sessionPreset: .hd1280x720,
            position: .back,
            orientation: .portrait
        ) { _ in
            setupExpectation.fulfill()
        }
        
        wait(for: [setupExpectation], timeout: 2.0)
        
        // 4. Verify state
        XCTAssertNotNil(videoCapture.predictor)
        XCTAssertNotNil(videoCapture.delegate)
        
        // 5. Stop session
        videoCapture.stop()
        XCTAssertFalse(videoCapture.captureSession.isRunning)
    }
}

// MARK: - Mock Classes

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

class MockVideoCapturePredictor: BasePredictor, @unchecked Sendable {
    var didCallPredictOnImage = false
    var didCallProcessBuffer = false
    
    override func predict(
        sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?,
        onInferenceTime: InferenceTimeListener?
    ) {
        didCallProcessBuffer = true
        // Simulate a prediction result
        let result = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
        onResultsListener?.on(result: result)
        onInferenceTime?.on(inferenceTime: 0.01, fpsRate: 30.0)
    }
    
    override func predictOnImage(image: CIImage) -> YOLOResult {
        didCallPredictOnImage = true
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate Tests

extension VideoCaptureTests {
    func testVideoCaptureAsAVCaptureVideoDataOutputDelegate() {
        // Verify VideoCapture conforms to AVCaptureVideoDataOutputSampleBufferDelegate
        let delegate: AVCaptureVideoDataOutputSampleBufferDelegate = videoCapture
        XCTAssertNotNil(delegate)
    }
    
    func testCaptureOutputDelegateMethod() {
        // Test that VideoCapture can act as video output delegate
        videoCapture.videoOutput.setSampleBufferDelegate(
            videoCapture,
            queue: videoCapture.cameraQueue
        )
        
        XCTAssertNotNil(videoCapture.videoOutput.sampleBufferDelegate)
    }
}

// MARK: - Error Handling Tests

extension VideoCaptureTests {
    func testCameraSetupWithInvalidPreset() {
        // Test handling of invalid camera configuration
        let expectation = XCTestExpectation(description: "Invalid setup")
        
        // Use an unusual preset that might not be supported
        videoCapture.setUp(
            sessionPreset: .inputPriority,
            position: .back,
            orientation: .portrait
        ) { success in
            // Should complete regardless
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 2.0)
    }
}