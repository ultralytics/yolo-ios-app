// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app and defines the registry of downloadable YOLO models.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  remoteModelsInfo maps task names (detection, segmentation, semantic segmentation, depth, classification, pose, OBB) to the
//  YOLO models available for download from the GitHub release. Entries listed here that are not already bundled locally
//  appear as downloadable options in the UI; ModelDownloadManager fetches and installs them on demand.

import Foundation

/// Maps task names to the YOLO models available for download, with their archive URLs.
public let remoteModelsInfo: [String: [(modelName: String, downloadURL: URL)]] = {
  let base = "https://github.com/ultralytics/yolo-ios-app/releases/download/models-v1.0.0"
  let sizes = ["n", "s", "m", "l", "x"]
  let tasks = [
    ("Detect", ""), ("Segment", "-seg"), ("Semantic", "-sem"), ("Depth", "-depth"),
    ("Classify", "-cls"), ("Pose", "-pose"), ("OBB", "-obb"),
  ]
  return tasks.reduce(into: [:]) { result, task in
    result[task.0] = sizes.map { size in
      let model = "yolo26\(size)\(task.1)"
      return (model, URL(string: "\(base)/\(model).mlpackage.zip")!)
    }
  }
}()
