// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app, defining remotely available YOLO models.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The RemoteModels file defines a registry of downloadable YOLO models available for each task type.
//  It provides a structured mapping between task categories (detection, segmentation, classification, etc.)
//  and the available models with their remote download URLs. These models are presented to users in the
//  application interface, allowing them to download and use additional models beyond those bundled with
//  the application. The dictionary structure enables easy filtering of models by task type and provides
//  all necessary information for the ModelDownloadManager to retrieve and install the models.

import Foundation

/// A dictionary mapping task names to available remote models with their download URLs.
public let remoteModelsInfo: [String: [(modelName: String, downloadURL: URL)]] = {
  let base = "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"
  let sizes = ["n", "s", "m", "l", "x"]
  let tasks = [
    ("Detect", ""), ("Segment", "-seg"), ("Classify", "-cls"), ("Pose", "-pose"), ("OBB", "-obb"),
  ]
  
  // Combine yolo11 and yolo26 models
  var result: [String: [(modelName: String, downloadURL: URL)]] = [:]
  
  for task in tasks {
    var models: [(modelName: String, downloadURL: URL)] = []
    
    // Add yolo26 models first (prioritized - they're newer and don't need NMS)
    for size in sizes {
      let model = "yolo26\(size)\(task.1)"
      models.append((model, URL(string: "\(base)/\(model).mlpackage.zip")!))
    }
    
    // Add yolo11 models (legacy support)
    for size in sizes {
      let model = "yolo11\(size)\(task.1)"
      models.append((model, URL(string: "\(base)/\(model).mlpackage.zip")!))
    }
    
    result[task.0] = models
  }
  
  return result
}()
