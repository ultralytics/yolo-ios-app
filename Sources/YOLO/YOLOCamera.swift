import SwiftUI
import AVFoundation

public struct YOLOCamera: View {
    @State private var yoloResult: YOLOResult?
    
    public let modelPathOrName: String
    public let task: YOLOTask
    public let cameraPosition: AVCaptureDevice.Position
    
    public init(
        modelPathOrName: String,
        task: YOLOTask = .detect,
        cameraPosition: AVCaptureDevice.Position = .back
    ) {
        self.modelPathOrName = modelPathOrName
        self.task = task
        self.cameraPosition = cameraPosition
    }
    
    
    public var body: some View {
        YOLOViewRepresentable(
            modelPathOrName: modelPathOrName,
            task: task,
            cameraPosition: cameraPosition
        ) { result in
            self.yoloResult = result
        }
    }
}

struct YOLOViewRepresentable: UIViewRepresentable {
    let modelPathOrName: String
    let task: YOLOTask
    let cameraPosition: AVCaptureDevice.Position
    let onDetection: ((YOLOResult) -> Void)?
    
    func makeUIView(context: Context) -> YOLOView {
        let yoloView = YOLOView(
            frame: .zero,
            modelPathOrName: modelPathOrName,
            task: task
        )
        return yoloView
    }
    
    func updateUIView(_ uiView: YOLOView, context: Context) {
        uiView.onDetection = onDetection
    }
}
