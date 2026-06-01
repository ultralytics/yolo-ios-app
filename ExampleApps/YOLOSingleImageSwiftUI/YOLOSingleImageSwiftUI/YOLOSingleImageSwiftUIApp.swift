// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing the SwiftUI app entry point for single
//  image object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://www.ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  YOLOSingleImageSwiftUIApp is the entry point for the SwiftUI single image object detection example. It initializes
//  the window group and sets ContentView as the root view, following the standard SwiftUI app lifecycle with the @main
//  attribute. The app shows how to integrate YOLO models for still image analysis in SwiftUI, letting users select
//  images from their photo library for processing.

import SwiftUI

/// The main entry point for the YOLO single image analysis SwiftUI example app.
///
/// Defines the app's scene structure with a single window group containing the ContentView, which provides the user
/// interface for selecting and analyzing images with YOLO models.
///
/// - Note: This app requires at least iOS 16.0 to run due to PhotosPicker API requirements.
/// - Important: Include proper photo library privacy descriptions in Info.plist.
@main
struct YOLOSingleImageSwiftUIApp: App {
  var body: some Scene {
    WindowGroup {
      ContentView()
    }
  }
}
