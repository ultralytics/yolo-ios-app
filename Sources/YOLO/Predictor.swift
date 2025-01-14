import Vision
import CoreImage

protocol ResultsListener {
    func on(result: YOLOResult)
}

protocol InferenceTimeListener {
    func on(inferenceTime: Double)
}

protocol FpsRateListener {
    func on(fpsRate: Double)
}

protocol Predictor{
    func predict(sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?, onInferenceTime: InferenceTimeListener?, onFpsRate: FpsRateListener?)
    func predictOnImage(image: CIImage) -> YOLOResult
    var labels: [String] { get set }
}

enum PredictorError: Error{
    case invalidTask
    case noLabelsFound
    case invalidUrl
    case modelFileNotFound
}
