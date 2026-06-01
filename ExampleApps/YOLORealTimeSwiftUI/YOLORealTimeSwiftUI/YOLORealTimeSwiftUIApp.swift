// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing the SwiftUI app entry point for
//  real-time object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://www.ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  YOLORealTimeSwiftUIApp is the entry point for the SwiftUI real-time object detection example. It initializes the
//  window group and sets ContentView as the root view, following the standard SwiftUI app lifecycle with the @main
//  attribute. The app shows how to integrate YOLO models for continuous object detection in a SwiftUI application.

import SwiftUI

/// The main entry point for the YOLO real-time detection SwiftUI example app.
///
/// Defines the app's scene structure with a single window group containing the ContentView, which provides real-time
/// camera-based object detection using the YOLOCamera component.
///
/// - Note: This app requires camera permissions in Info.plist and at least iOS 16.0 to run.
/// - Important: The camera requires user permission, which will be requested at runtime.
@main
struct YOLORealTimeSwiftUIApp: App {
  var body: some Scene {
    WindowGroup {
      ContentView()
    }
  }
}
