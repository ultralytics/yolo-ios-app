import SwiftUI
import YOLO

struct ContentView: View {
    var body: some View {
        YOLOCamera(
            modelPathOrName: "yolov8n-pose",
            task: .pose,
            cameraPosition: .back
        )
        .ignoresSafeArea()
    }
}

#Preview {
    ContentView()
}
