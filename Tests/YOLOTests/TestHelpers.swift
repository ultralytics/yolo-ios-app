// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import Vision
import CoreML
@testable import YOLO

// MARK: - Mock Classes

class MockResultsListener: ResultsListener {
    var onResultHandler: ((YOLOResult) -> Void)?
    
    func on(result: YOLOResult) {
        onResultHandler?(result)
    }
}

class MockInferenceTimeListener: InferenceTimeListener {
    var onInferenceTimeHandler: ((Double, Double) -> Void)?
    
    func on(inferenceTime: Double, fpsRate: Double) {
        onInferenceTimeHandler?(inferenceTime, fpsRate)
    }
}

// MARK: - Mock VNRequest Classes

class MockVNRequestWithResults: VNRequest, @unchecked Sendable {
    private var mockResults: [Any]?
    
    init(results: [Any]) {
        super.init(completionHandler: nil)
        self.mockResults = results
    }
    
    override var results: [VNObservation]? {
        // Return VNObservation types only
        return mockResults?.compactMap { $0 as? VNObservation }
    }
}

// We cannot properly subclass VNCoreMLFeatureValueObservation
// Create a simple mock that won't be used as VNObservation
class MockFeatureValueObservation {
    let multiArray: MLMultiArray
    
    init(multiArray: MLMultiArray) {
        self.multiArray = multiArray
    }
    
    var featureValue: MLFeatureValue {
        return MLFeatureValue(multiArray: multiArray)
    }
}

// We'll use MockVNRequestWithResults instead of MockVNCoreMLRequest
// since VNCoreMLRequest requires a valid model which is difficult to mock

class MockVNRecognizedObjectObservation: VNRecognizedObjectObservation {
    private let mockBoundingBox: CGRect
    private let mockLabels: [VNClassificationObservation]
    
    init(boundingBox: CGRect, labels: [VNClassificationObservation]) {
        self.mockBoundingBox = boundingBox
        self.mockLabels = labels
        super.init()
    }
    
    override var boundingBox: CGRect {
        return mockBoundingBox
    }
    
    override var labels: [VNClassificationObservation] {
        return mockLabels
    }
    
    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

class MockVNClassificationObservation: VNClassificationObservation {
    private let mockIdentifier: String
    private let mockConfidence: VNConfidence
    
    init(identifier: String, confidence: VNConfidence) {
        self.mockIdentifier = identifier
        self.mockConfidence = confidence
        super.init()
    }
    
    override var identifier: String {
        return mockIdentifier
    }
    
    override var confidence: VNConfidence {
        return mockConfidence
    }
    
    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}