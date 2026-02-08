// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreMedia
import UIKit
import YOLO

/// Manager for external display communication
class ExternalDisplayManager {
  static let shared = ExternalDisplayManager()

  private init() {}

  /// Posts YOLO results for external display
  func shareResults(_ results: YOLOResult) {
    NotificationCenter.default.post(
      name: .yoloResultsAvailable,
      object: nil,
      userInfo: ["results": results]
    )
  }

  /// Posts model change notification with task type and model name
  func notifyModelChange(task: YOLOTask, modelName: String) {
    let taskString = String(describing: task).lowercased()

    NotificationCenter.default.post(
      name: .modelDidChange,
      object: nil,
      userInfo: [
        "task": taskString,
        "modelName": modelName,
      ]
    )
  }
}
