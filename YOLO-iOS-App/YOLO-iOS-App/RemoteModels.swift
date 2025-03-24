//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
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
public let remoteModelsInfo: [String: [(modelName: String, downloadURL: URL)]] = [
  "Detect": [
    //        ("yolo11n",  URL(string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11n.mlpackage.zip")!),
    //        ("yolo11s",  URL(string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11s.mlpackage.zip")!),
    //        ("yolo11m",  URL(string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11m.mlpackage.zip")!),
    (
      "yolo11l",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11l.mlpackage.zip"
      )!
    ),
    (
      "yolo11x",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11x.mlpackage.zip"
      )!
    ),
  ],
  "Segment": [
    //        ("yolo11n-seg",  URL(string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11n-seg.mlpackage.zip")!),
    (
      "yolo11s-seg",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11s-seg.mlpackage.zip"
      )!
    ),
    (
      "yolo11m-seg",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11m-seg.mlpackage.zip"
      )!
    ),
    (
      "yolo11l-seg",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11l-seg.mlpackage.zip"
      )!
    ),
    (
      "yolo11x-seg",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11x-seg.mlpackage.zip"
      )!
    ),
  ],
  "Classify": [
    //        ("yolo11n-cls", URL(string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11n-clswasq5.mlpackage.zip")!),
    (
      "yolo11s-cls",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11s-cls.mlpackage.zip"
      )!
    ),
    (
      "yolo11m-cls",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11m-cls.mlpackage.zip"
      )!
    ),
    (
      "yolo11l-cls",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11l-cls.mlpackage.zip"
      )!
    ),
    (
      "yolo11x-cls",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11x-cls.mlpackage.zip"
      )!
    ),
  ],
  "Pose": [
    //        ("yolo11n-pose", URL(string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11n-pose.mlpackage.zip")!),
    (
      "yolo11s-pose",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11s-pose.mlpackage.zip"
      )!
    ),
    (
      "yolo11m-pose",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11m-pose.mlpackage.zip"
      )!
    ),
    (
      "yolo11l-pose",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11l-pose.mlpackage.zip"
      )!
    ),
    (
      "yolo11x-pose",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11x-pose.mlpackage.zip"
      )!
    ),
  ],
  "Obb": [
    //        ("yolo11n-obb",  URL(string: "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11n-obb.mlpackage.zip")!),
    (
      "yolo11s-obb",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11s-obb.mlpackage.zip"
      )!
    ),
    (
      "yolo11m-obb",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11m-obb.mlpackage.zip"
      )!
    ),
    (
      "yolo11l-obb",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11l-obb.mlpackage.zip"
      )!
    ),
    (
      "yolo11x-obb",
      URL(
        string:
          "https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/yolo11x-obb.mlpackage.zip"
      )!
    ),
  ],
]
