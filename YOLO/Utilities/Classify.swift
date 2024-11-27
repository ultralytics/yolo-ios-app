//
//  Classify.swift
//  YOLO
//

import Foundation
import UIKit
import Vision

extension ViewController {
    // view
    func setupClassifyOverlay() {

        classifyOverlay = UILabel(frame: CGRect(x: view.center.x - 100, y: view.center.y - 50, width: 200, height: 100))
        
        classifyOverlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        classifyOverlay.clipsToBounds = true
        classifyOverlay.layer.cornerRadius = 8
        classifyOverlay.numberOfLines = 2
        classifyOverlay.textAlignment = .left
        view.addSubview(classifyOverlay)
        classifyOverlay.isHidden = true
    }
    
    func showClassifyUI() {
        taskSegmentControl.selectedSegmentIndex = 1
        modelSegmentedControl.selectedSegmentIndex = 0
        classifyOverlay.isHidden = false
    }
    
    func updateClassifyOverlay() {
        
        classifyOverlay.frame = CGRect(x: view.center.x - 100, y: view.center.y - 50, width: 200, height: 100)
    }
    // post process
    
    func postProcessClassify(request: VNRequest) {
        if let observation = visionRequest.results as? [VNCoreMLFeatureValueObservation]{
            
            // Get the MLMultiArray from the observation
            let multiArray = observation.first?.featureValue.multiArrayValue
            
            if let multiArray = multiArray {
                // Initialize an array to store the classes
                var valuesArray = [Double]()
                
                // Loop through the MLMultiArray and append its values to the array
                for i in 0..<multiArray.count {
                    let value = multiArray[i].doubleValue
                    valuesArray.append(value)
                }
                
                // Create an indexed map as a dictionary
                var indexedMap = [Int: Double]()
                for (index, value) in valuesArray.enumerated() {
                    indexedMap[index] = value
                }
                
                // Sort the dictionary in descending order based on values
                let sortedMap = indexedMap.sorted(by: { $0.value > $1.value })
                
                var recognitions: [[String:Any]] = []
                for (index, value) in sortedMap {
                    let label = self.classifyLabels[index]
                    recognitions.append(["label": label,
                                         "confidence": value,
                                         "index": index])
                }
                print(recognitions)
            }
        } else if let observations = request.results as? [VNClassificationObservation] {
            
            var recognitions: [[String: Any]] = []
            
            // Convert each VNClassificationObservation into the desired format
            guard let topResult = observations.first else { return }
            let label = topResult.identifier // Class label
            let confidence = topResult.confidence // Confidence score (between 0 and 1)
            let percentageValue = confidence * 100
            let formattedPercentage = round(percentageValue * 10) / 10

            let resultText = "  \(label)\n  \(formattedPercentage) %"
            DispatchQueue.main.async {
                self.classifyOverlay.text = resultText
            }
            
        }
    }
}
