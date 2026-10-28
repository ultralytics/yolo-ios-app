// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app and defines the registry of downloadable YOLO models.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  remoteModelsInfo maps task names (detection, segmentation, semantic segmentation, depth, classification, pose, OBB) to the
//  YOLO models available for download from the GitHub release. Entries listed here that are not already bundled locally
//  appear as downloadable options in the UI; ModelDownloadManager fetches and installs them on demand.

import Foundation
import UltralyticsYOLO

/// The format the Core AI setting asks for: Core ML (`.mlpackage`) unless the user turned on Settings → Ultralytics
/// YOLO → Core AI Models on a device that can run Core AI (iOS 27 and later).
var preferredModelExtension: String {
  UserDefaults.standard.bool(forKey: "use_core_ai") && BasePredictor.isCoreAIAvailable
    ? "aimodel" : "mlpackage"
}

/// The official asset format the app lists, downloads and caches. It follows `preferredModelExtension` only when the
/// app relists its models (at launch, and on returning to the foreground with the setting changed), so a list, its
/// downloads and its cache paths always agree.
var remoteModelExtension = preferredModelExtension

/// Maps task names to the YOLO models available for download, with their archive URLs.
public var remoteModelsInfo: [String: [(modelName: String, downloadURL: URL)]] {
  let base = "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"
  let sizes = ["n", "s", "m", "l", "x"]
  let tasks = [
    ("Detect", ""), ("Segment", "-seg"), ("Semantic", "-sem"), ("Depth", "-depth"),
    ("Classify", "-cls"), ("Pose", "-pose"), ("OBB", "-obb"),
  ]
  return tasks.reduce(into: [:]) { result, task in
    result[task.0] = sizes.map { size in
      let model = "yolo26\(size)\(task.1)"
      return (model, URL(string: "\(base)/\(model).\(remoteModelExtension).zip")!)
    }
  }
}
