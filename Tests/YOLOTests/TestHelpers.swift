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
    private var mockResults: [VNObservation]?
    
    init(results: [Any]) {
        super.init(completionHandler: nil)
        // Convert Any results to VNObservation
        self.mockResults = results.compactMap { $0 as? VNObservation }
    }
    
    override var results: [VNObservation]? {
        return mockResults
    }
}

class MockVNCoreMLFeatureValueObservation: VNCoreMLFeatureValueObservation, @unchecked Sendable {
    let multiArray: MLMultiArray
    
    init(multiArray: MLMultiArray) {
        self.multiArray = multiArray
        super.init()
    }
    
    override var featureValue: MLFeatureValue {
        return MLFeatureValue(multiArray: multiArray)
    }
    
    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

class MockVNCoreMLRequest: VNCoreMLRequest, @unchecked Sendable {
    private var mockResults: [VNObservation]?
    
    init(results: [Any]) {
        // Create a dummy model for initialization
        let config = MLModelConfiguration()
        if let dummyModel = try? MLModel(contentsOf: Bundle.main.bundleURL, configuration: config),
           let vncoreModel = try? VNCoreMLModel(for: dummyModel) {
            super.init(model: vncoreModel)
        } else {
            // If we can't create a dummy model, we need to handle this differently
            // This is a limitation of testing CoreML requests
            fatalError("Cannot create mock VNCoreMLRequest without a valid model")
        }
        self.mockResults = results.compactMap { $0 as? VNObservation }
    }
    
    override var results: [VNObservation]? {
        return mockResults
    }
    
    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

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