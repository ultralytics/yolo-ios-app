import Foundation
import Vision
import UIKit

public class BasePredictor : Predictor {
    var detector: VNCoreMLModel!
    var visionRequest: VNCoreMLRequest?
    public var labels = [String]()
    var currentBuffer: CVPixelBuffer?
    weak var currentOnResultsListener: ResultsListener?
    weak var currentOnInferenceTimeListener: InferenceTimeListener?
    var inputSize: CGSize!
    var modelInputSize : (width: Int, height: Int) = (0,0)

    var t0 = 0.0  // inference start
    var t1 = 0.0  // inference dt
    var t2 = 0.0  // inference dt smoothed
    var t3 = CACurrentMediaTime()  // FPS start
    var t4 = 0.0  // FPS dt smoothed
    public var isUpdating: Bool = false
    
    init(unwrappedModelURL: URL) {
        
        let ext = unwrappedModelURL.pathExtension.lowercased()
        let isCompiled = (ext == "mlmodelc")
        let config = MLModelConfiguration()
        config.setValue(1, forKey: "experimentalMLE5EngineUsage")
        var mlModel: MLModel
        do {
            if isCompiled {
                mlModel = try MLModel(contentsOf: unwrappedModelURL,configuration: config)
            } else {
                let compiledUrl = try MLModel.compileModel(at: unwrappedModelURL)
                mlModel = try MLModel(contentsOf: compiledUrl,configuration: config)
            }
        } catch {
            fatalError(PredictorError.modelFileNotFound.localizedDescription)
        }
        
        guard let userDefined = mlModel.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]
        else { return }
        
        var allLabels: String = ""
        if let labelsData = userDefined["classes"] {
            allLabels = labelsData
            labels = allLabels.components(separatedBy: ",")
        } else if let labelsData = userDefined["names"] {
            // Remove curly braces and spaces from the input string
            let cleanedInput = labelsData.replacingOccurrences(of: "{", with: "")
                .replacingOccurrences(of: "}", with: "")
                .replacingOccurrences(of: " ", with: "")
            
            // Split the cleaned string into an array of key-value pairs
            let keyValuePairs = cleanedInput.components(separatedBy: ",")
            
            
            for pair in keyValuePairs {
                // Split each key-value pair into key and value
                let components = pair.components(separatedBy: ":")
                
                
                // Check if we have at least two components
                if components.count >= 2 {
                    // Get the second component and trim any leading/trailing whitespace
                    let extractedString = components[1].trimmingCharacters(in: .whitespaces)
                    
                    // Remove single quotes if they exist
                    let cleanedString = extractedString.replacingOccurrences(of: "'", with: "")
                    
                    labels.append(cleanedString)
                } else {
                    print("Invalid input string")
                }
            }
            
        } else {
            fatalError("Invalid metadata format")
        }
        
        modelInputSize = getModelInputSize(for: mlModel)
        
        detector = try! VNCoreMLModel(for: mlModel)
        detector.featureProvider = ThresholdProvider()
        
        visionRequest = {
            let request = VNCoreMLRequest(model: detector, completionHandler: {
                [weak self] request, error in
                self?.processObservations(for: request, error: error)
            })
            request.imageCropAndScaleOption = .scaleFill
            return request
        }()
    }
    
    func predict(sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?, onInferenceTime: InferenceTimeListener?) {
        if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            inputSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
            currentOnResultsListener = onResultsListener
            currentOnInferenceTimeListener = onInferenceTime
//            currentOnFpsRateListener = onFpsRate
            
            /// - Tag: MappingOrientation
            // The frame is always oriented based on the camera sensor,
            // so in most cases Vision needs to rotate it for the model to work as expected.
            let imageOrientation: CGImagePropertyOrientation = .up
            
            // Invoke a VNRequestHandler with that image
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
            t0 = CACurrentMediaTime()  // inference start
            do {
                if(visionRequest != nil){
                    try handler.perform([visionRequest!])
                }
            } catch {
                print(error)
            }
            t1 = CACurrentMediaTime() - t0  // inference dt
            
            
            currentBuffer = nil
        }
    }
    
    var confidenceThreshold = 0.25
    func setConfidenceThreshold(confidence: Double) {
        confidenceThreshold = confidence
    }
    
    var iouThreshold = 0.4
    func setIouThreshold(iou: Double){
        iouThreshold = iou
    }
    
    var numItemsThreshold = 30
    func setNumItemsThreshold(numItems: Int){
        numItemsThreshold = numItems
    }
    
    func processObservations(for request: VNRequest, error: Error?) {
        
    }
    
    func predictOnImage(image: CIImage) -> YOLOResult {
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }

    func getModelInputSize(for model: MLModel) -> (width: Int, height: Int) {
        guard let inputDescription = model.modelDescription.inputDescriptionsByName.first?.value else {
            print("can not find input description")
            return (0,0)
        }
        
        if let multiArrayConstraint = inputDescription.multiArrayConstraint {
            let shape = multiArrayConstraint.shape
            if shape.count >= 2 {
                let height = shape[0].intValue
                let width = shape[1].intValue
                return (width: width, height: height)
            }
        }
        
        // 入力仕様がImageの場合
        if let imageConstraint = inputDescription.imageConstraint {
            let width = Int(imageConstraint.pixelsWide)
            let height = Int(imageConstraint.pixelsHigh)
            return (width: width, height: height)
        }
        
        print("an not find input size")
        return (0,0)
    }
}
