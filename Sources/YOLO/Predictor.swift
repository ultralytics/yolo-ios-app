import Vision
import CoreImage

protocol ResultsListener: AnyObject {
    func on(result: YOLOResult)
}

protocol InferenceTimeListener: AnyObject {
    func on(inferenceTime: Double,fpsRate: Double)
}
//
//protocol FpsRateListener: AnyObject {
//    func on(fpsRate: Double)
//}

protocol Predictor{
    func predict(sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?, onInferenceTime: InferenceTimeListener?)
    func predictOnImage(image: CIImage) -> YOLOResult
    var labels: [String] { get set }
    var isUpdating: Bool { get set }
}

enum PredictorError: Error{
    case invalidTask
    case noLabelsFound
    case invalidUrl
    case modelFileNotFound
}
