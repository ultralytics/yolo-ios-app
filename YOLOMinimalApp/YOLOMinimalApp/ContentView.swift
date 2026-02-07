// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI

/// Root view with tabs for real-time camera and single-image modes.
struct ContentView: View {
  var body: some View {
    TabView {
      CameraView()
        .tabItem {
          Label("Camera", systemImage: "camera.fill")
        }

      PhotoView()
        .tabItem {
          Label("Photo", systemImage: "photo.fill")
        }
    }
  }
}
