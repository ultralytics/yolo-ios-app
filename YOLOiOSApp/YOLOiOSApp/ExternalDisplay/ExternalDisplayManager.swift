// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreMedia
import UIKit
import YOLO

/// Manager for external display communication
class ExternalDisplayManager {
  static let shared = ExternalDisplayManager()
  private static let dedicatedModeKey = "dedicated_external_display"

  private init() {}

  static func registerDefaults() {
    UserDefaults.standard.register(defaults: [dedicatedModeKey: true])
  }

  static var isDedicatedModeEnabled: Bool {
    UserDefaults.standard.synchronize()
    return UserDefaults.standard.object(forKey: dedicatedModeKey) as? Bool ?? true
  }

  static func refreshModeIfNeeded() {
    let externalSessions = UIApplication.shared.openSessions.filter { $0.role.isExternalDisplay }

    if !isDedicatedModeEnabled {
      NotificationCenter.default.post(name: .externalDisplayDisconnected, object: nil)
      externalSessions.forEach {
        UIApplication.shared.requestSceneSessionDestruction($0, options: nil, errorHandler: nil)
      }
      return
    }

    guard externalSessions.contains(where: { $0.configuration.delegateClass == nil }) else {
      return
    }

    externalSessions.forEach {
      UIApplication.shared.requestSceneSessionDestruction($0, options: nil, errorHandler: nil)
    }
    if #available(iOS 17.0, *) {
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        guard isDedicatedModeEnabled else { return }
        let role = UISceneSession.Role(
          rawValue: "UIWindowSceneSessionRoleExternalDisplayNonInteractive")
        let request = UISceneSessionActivationRequest(role: role)
        UIApplication.shared.activateSceneSession(for: request, errorHandler: nil)
      }
    }
  }

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

extension UISceneSession.Role {
  var isExternalDisplay: Bool {
    rawValue == "UIWindowSceneSessionRoleExternalDisplay"
      || rawValue == "UIWindowSceneSessionRoleExternalDisplayNonInteractive"
  }
}
