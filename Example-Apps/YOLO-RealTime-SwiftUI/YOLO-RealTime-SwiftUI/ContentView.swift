import SwiftUI
import YOLO

struct ContentView: View {
  var body: some View {
    YOLOCamera(
      modelPathOrName: "yolo11n-obb",
      task: .obb,
      cameraPosition: .back
    )
    .ignoresSafeArea()
  }
}

#Preview {
  ContentView()
}
