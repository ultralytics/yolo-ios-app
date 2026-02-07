// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI

/// Main entry point for the Ultralytics YOLO iOS App.
@main
struct YOLOiOSApp: App {
  init() {
    UIApplication.shared.isIdleTimerDisabled = true
    UIDevice.current.isBatteryMonitoringEnabled = true
  }

  var body: some Scene {
    WindowGroup {
      MainView()
        .preferredColorScheme(.dark)
    }
  }
}
