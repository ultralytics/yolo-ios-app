import SwiftUI
import YOLO

struct ContentView: View {
    var body: some View {
        YOLOCamera(
            modelPathOrName: "yolov8n-seg",
            task: .segment,
            cameraPosition: .back
        )
        .ignoresSafeArea()
    }
}

#Preview {
    ContentView()
}
