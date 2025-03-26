//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing the SwiftUI app entry point for real-time object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLORealTimeSwiftUIApp serves as the main application entry point for the SwiftUI-based
//  real-time object detection example. It initializes the application's window group and sets
//  ContentView as the root view. This structure follows the standard SwiftUI app lifecycle
//  pattern, where the @main attribute designates this struct as the application's entry point.
//  The app demonstrates how to integrate YOLO models for continuous object detection in a SwiftUI
//  application, providing a simple but effective implementation that developers can build upon.

import SwiftUI

/// The main entry point for the YOLO real-time detection SwiftUI example app.
///
/// This struct represents the application instance and defines the app's scene structure.
/// It creates a single window group containing the ContentView, which provides real-time
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
