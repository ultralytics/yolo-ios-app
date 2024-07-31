//
//  ModelURLs.swift
//  YOLO
//
//  Created by 間嶋大輔 on 2024/07/28.
//  Copyright © 2024 Ultralytics. All rights reserved.
//

import Foundation

let presetModels = ["yolov8n","yolov8s","yolov8m","yolov8l","yolov8x"]

let fileMappings = [
    ("yolov8n", URL(string: "https://firebasestorage.googleapis.com/v0/b/sincere-nirvana-292404.appspot.com/o/yolov8n.mlpackage.zip?alt=media&token=80a62f8e-96f1-4355-8857-51b4ecd6f2e3")!),
    ("yolov8s", URL(string: "https://firebasestorage.googleapis.com/v0/b/sincere-nirvana-292404.appspot.com/o/yolov8s.mlpackage.zip?alt=media&token=8f18404a-6c33-49f6-b89a-a3b69a4aa0bb")!),
    ("yolov8l", URL(string: "https://firebasestorage.googleapis.com/v0/b/sincere-nirvana-292404.appspot.com/o/yolov8l.mlpackage.zip?alt=media&token=393c644e-b848-4c97-a4d0-61cfa6ffc307")!),
    ("yolov8x", URL(string: "https://firebasestorage.googleapis.com/v0/b/sincere-nirvana-292404.appspot.com/o/yolov8x.mlpackage.zip?alt=media&token=f3074450-00bd-4b44-962d-f4e122d73e6b")!)
]
