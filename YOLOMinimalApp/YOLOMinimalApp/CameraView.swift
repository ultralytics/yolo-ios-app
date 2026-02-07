// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import SwiftUI
import YOLOUI

/// Real-time camera view with task picker for all 5 YOLO tasks.
struct CameraView: View {
  @State private var selectedTask: YOLOTask = .detect

  var body: some View {
    ZStack(alignment: .top) {
      YOLOCamera(model: modelName, task: selectedTask)
        .ignoresSafeArea()

      // Task picker overlay
      Picker("Task", selection: $selectedTask) {
        Text("Detect").tag(YOLOTask.detect)
        Text("Segment").tag(YOLOTask.segment)
        Text("Classify").tag(YOLOTask.classify)
        Text("Pose").tag(YOLOTask.pose)
        Text("OBB").tag(YOLOTask.obb)
      }
      .pickerStyle(.segmented)
      .padding(.horizontal)
      .padding(.top, 8)
    }
  }

  private var modelName: String {
    switch selectedTask {
    case .detect: return "yolo26n"
    case .segment: return "yolo26n-seg"
    case .classify: return "yolo26n-cls"
    case .pose: return "yolo26n-pose"
    case .obb: return "yolo26n-obb"
    }
  }
}
