// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation
import CoreImage
import AVFoundation
import UIKit
@testable import YOLO

/// A simple mock predictor that implements the Predictor protocol for testing
/// This avoids inheritance issues and provides a clean testing interface
class MockPredictor: NSObject, Predictor {
    // Test control properties
    var didCallPredict = false
    var didCallPredictOnImage = false
    var predictCallCount = 0
    
    // Configurable test data
    var mockResult: YOLOResult = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    var mockInferenceTime: Double = 0.01
    var mockFpsRate: Double = 30.0
    
    // Predictor protocol requirements
    var labels: [String] = []
    var isUpdating: Bool = false
    
    func predict(
        sampleBuffer: CMSampleBuffer,
        onResultsListener: ResultsListener?,
        onInferenceTime: InferenceTimeListener?
    ) {
        didCallPredict = true
        predictCallCount += 1
        
        // Return configured mock data
        onResultsListener?.on(result: mockResult)
        onInferenceTime?.on(inferenceTime: mockInferenceTime, fpsRate: mockFpsRate)
    }
    
    func predictOnImage(image: CIImage) -> YOLOResult {
        didCallPredictOnImage = true
        return mockResult
    }
}

/// A test double for VideoCapture that doesn't rely on actual camera hardware
class TestableVideoCapture: VideoCapture {
    // Override properties for testing
    var mockPredictor: Predictor?
    
    override var predictor: Predictor? {
        get { return mockPredictor }
        set { mockPredictor = newValue }
    }
    
    // Skip actual camera setup in tests
    override func setUp(
        sessionPreset: AVCaptureSession.Preset,
        position: AVCaptureDevice.Position,
        orientation: UIDeviceOrientation,
        completion: @escaping (Bool) -> Void
    ) {
        // Simulate successful setup without actual camera
        DispatchQueue.main.async {
            completion(true)
        }
    }
    
    override func start() {
        // No-op for tests
    }
    
    override func stop() {
        // No-op for tests
    }
}