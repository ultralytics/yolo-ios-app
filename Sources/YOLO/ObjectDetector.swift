import Foundation
import Vision
import UIKit

class ObjectDetector: Predictor {
    private var detector: VNCoreMLModel!
    private var visionRequest: VNCoreMLRequest?
    private var currentBuffer: CVPixelBuffer?
    private var currentOnResultsListener: ResultsListener?
    private var currentOnInferenceTimeListener: InferenceTimeListener?
    private var currentOnFpsRateListener: FpsRateListener?
    private var inputSize: CGSize!
    public var labels = [String]()
    var t0 = 0.0  // inference start
    var t1 = 0.0  // inference dt
    var t2 = 0.0  // inference dt smoothed
    var t3 = CACurrentMediaTime()  // FPS start
    var t4 = 0.0  // FPS dt smoothed
    private var isSync = false
    
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
    
    func predict(sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?, onInferenceTime: InferenceTimeListener?, onFpsRate: FpsRateListener?) {
        if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            inputSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
            currentOnResultsListener = onResultsListener
            currentOnInferenceTimeListener = onInferenceTime
            currentOnFpsRateListener = onFpsRate
            
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
    
    private var confidenceThreshold = 0.2
    func setConfidenceThreshold(confidence: Double) {
        confidenceThreshold = confidence
        detector.featureProvider = ThresholdProvider(iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
    }
    
    private var iouThreshold = 0.4
    func setIouThreshold(iou: Double){
        iouThreshold = iou
        detector.featureProvider = ThresholdProvider(iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
    }
    
    private var numItemsThreshold = 30
    func setNumItemsThreshold(numItems: Int){
        numItemsThreshold = numItems
    }
    
    func processObservations(for request: VNRequest, error: Error?) {
            if let results = request.results as? [VNRecognizedObjectObservation] {
                var boxes = [Box]()

                for i in 0..<100 {
                    if i < results.count && i < self.numItemsThreshold {
                        let prediction = results[i]
                        let invertedBox = CGRect(x: prediction.boundingBox.minX, y: 1-prediction.boundingBox.maxY, width: prediction.boundingBox.width, height: prediction.boundingBox.height)
                        let imageRect = VNImageRectForNormalizedRect(invertedBox, Int(inputSize.width), Int(inputSize.height))
                        
                        // The labels array is a list of VNClassificationObservation objects,
                        // with the highest scoring class first in the list.
                        let label = prediction.labels[0].identifier
                        let index = self.labels.firstIndex(of: label) ?? 0
                        let confidence = prediction.labels[0].confidence
                        let box = Box(index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox)
                        boxes.append(box)
                    }
                }
                
                // Measure FPS
                if self.t1 < 10.0 {  // valid dt
                    self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
                }
                self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
                self.t3 = CACurrentMediaTime()

                self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000)  // t2 seconds to ms
                self.currentOnFpsRateListener?.on(fpsRate: 1 / self.t4)
                let result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: self.t2, fps: 1 / self.t4)

                self.currentOnResultsListener?.on(result: result)

            }
    }
    
    func predictOnImage(image: CIImage) -> YOLOResult {
        let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
        guard let request = visionRequest else {
            let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0)
            return emptyResult
        }
        var boxes = [Box]()
        
        let imageWidth = image.extent.width
        let imageHeight = image.extent.height
        self.inputSize = CGSize(width: imageWidth, height: imageHeight)
        
        do {
            try requestHandler.perform([request])
            if let results = request.results as? [VNRecognizedObjectObservation] {
                for i in 0..<100 {
                    if i < results.count && i < self.numItemsThreshold {
                        let prediction = results[i]
                        let invertedBox = CGRect(x: prediction.boundingBox.minX, y: 1-prediction.boundingBox.maxY, width: prediction.boundingBox.width, height: prediction.boundingBox.height)
                        let imageRect = VNImageRectForNormalizedRect(invertedBox, Int(inputSize.width), Int(inputSize.height))
                        
                        // The labels array is a list of VNClassificationObservation objects,
                        // with the highest scoring class first in the list.
                        let label = prediction.labels[0].identifier
                        let index = self.labels.firstIndex(of: label) ?? 0
                        let confidence = prediction.labels[0].confidence
                        let box = Box(index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox)
                        boxes.append(box)
                    }
                }
            }
        } catch {
            print(error)
        }
        let result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: t1)
        return result
    }
}
