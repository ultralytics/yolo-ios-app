//
//  VNOnnxRequest.swift
//  YOLO
//
//  Created by Pradeep Banavara on 24/03/24.
//  Copyright © 2024 Ultralytics. All rights reserved.
//

import Foundation
import CoreML
import Vision

class VNOnnxRequest {
    
    var mPath: String?
    var onnxResults: ORTValue?
    init(modelPath: String, completionHandler: (_ result: ORTValue) -> ()) {
        mPath = modelPath
        if onnxResults != nil {
            completionHandler(onnxResults!)
        }
        
    }
}
