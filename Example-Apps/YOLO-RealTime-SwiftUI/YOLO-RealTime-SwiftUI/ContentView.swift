import SwiftUI
import YOLO

struct ContentView: View {
    var body: some View {
        YOLOCamera(
            modelPathOrName: "yolo11n-cls",
            task: .classify,
            cameraPosition: .back
        )
        .ignoresSafeArea()
    }
}

#Preview {
    ContentView()
}
