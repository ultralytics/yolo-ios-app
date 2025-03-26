//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing the SwiftUI app entry point for single image object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOSingleImageSwiftUIApp serves as the main application entry point for the SwiftUI-based
//  single image object detection example. It initializes the application's window group and sets
//  ContentView as the root view. This structure follows the standard SwiftUI app lifecycle
//  pattern, where the @main attribute designates this struct as the application's entry point.
//  The app demonstrates how to integrate YOLO models for still image analysis in a SwiftUI
//  application, allowing users to select images from their photo library for processing.

import SwiftUI

/// The main entry point for the YOLO single image analysis SwiftUI example app.
///
/// This struct represents the application instance and defines the app's scene structure.
/// It creates a single window group containing the ContentView, which provides the user interface
/// for selecting and analyzing images with YOLO models.
///
/// - Note: This app requires at least iOS 16.0 to run due to PhotosPicker API requirements.
/// - Important: To use this app, you need to include proper privacy descriptions in Info.plist
///   for accessing the photo library.
@main
struct YOLOSingleImageSwiftUIApp: App {
  var body: some Scene {
    WindowGroup {
      ContentView()
    }
  }
}
