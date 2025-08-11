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

private let baseURL = "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"

/// Create URL for model download.
private func modelURL(_ name: String) -> URL {
  URL(string: "\(baseURL)/\(name).mlpackage.zip")!
}

/// A dictionary mapping task names to available remote models with their download URLs.
public let remoteModelsInfo: [String: [(modelName: String, downloadURL: URL)]] = [
  "Detect": [
    ("yolo11n", modelURL("yolo11n")),
    ("yolo11s", modelURL("yolo11s")),
    ("yolo11m", modelURL("yolo11m")),
    ("yolo11l", modelURL("yolo11l")),
    ("yolo11x", modelURL("yolo11x")),
  ],
  "Segment": [
    ("yolo11n-seg", modelURL("yolo11n-seg")),
    ("yolo11s-seg", modelURL("yolo11s-seg")),
    ("yolo11m-seg", modelURL("yolo11m-seg")),
    ("yolo11l-seg", modelURL("yolo11l-seg")),
    ("yolo11x-seg", modelURL("yolo11x-seg")),
  ],
  "Classify": [
    ("yolo11n-cls", modelURL("yolo11n-cls")),
    ("yolo11s-cls", modelURL("yolo11s-cls")),
    ("yolo11m-cls", modelURL("yolo11m-cls")),
    ("yolo11l-cls", modelURL("yolo11l-cls")),
    ("yolo11x-cls", modelURL("yolo11x-cls")),
  ],
  "Pose": [
    ("yolo11n-pose", modelURL("yolo11n-pose")),
    ("yolo11s-pose", modelURL("yolo11s-pose")),
    ("yolo11m-pose", modelURL("yolo11m-pose")),
    ("yolo11l-pose", modelURL("yolo11l-pose")),
    ("yolo11x-pose", modelURL("yolo11x-pose")),
  ],
  "OBB": [
    ("yolo11n-obb", modelURL("yolo11n-obb")),
    ("yolo11s-obb", modelURL("yolo11s-obb")),
    ("yolo11m-obb", modelURL("yolo11m-obb")),
    ("yolo11l-obb", modelURL("yolo11l-obb")),
    ("yolo11x-obb", modelURL("yolo11x-obb")),
  ],
]
