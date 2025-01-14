import SwiftUI
import YOLO

struct ContentView: View {
    var body: some View {
        YOLOCamera(
            modelPathOrName: "yolo11n",
            task: .detect,
            cameraPosition: .back
        )
        .ignoresSafeArea()
    }
}

#Preview {
    ContentView()
}
