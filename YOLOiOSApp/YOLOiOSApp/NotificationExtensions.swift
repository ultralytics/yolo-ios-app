// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation

// Notification names for external display events
extension Notification.Name {
    static let externalDisplayConnected = Notification.Name("ExternalDisplayConnected")
    static let externalDisplayDisconnected = Notification.Name("ExternalDisplayDisconnected")
    static let externalDisplayReady = Notification.Name("ExternalDisplayReady")
    static let externalDisplayUIToggle = Notification.Name("ExternalDisplayUIToggle")
    static let shareCameraSession = Notification.Name("ShareCameraSession")
    static let yoloResultsAvailable = Notification.Name("YOLOResultsAvailable")
    static let thresholdDidChange = Notification.Name("ThresholdDidChange")
    static let taskDidChange = Notification.Name("TaskDidChange")
}