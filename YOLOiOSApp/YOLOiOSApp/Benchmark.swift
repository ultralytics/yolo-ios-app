// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app and benchmarks the bundled models on the device.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  Launching the app with the `--benchmark` argument times every model bundled under `Models/<Task>/` (Core ML
//  `.mlpackage` and Core AI `.aimodel`, any head) with hardware acceleration and CPU only, then prints a markdown table.
//  It never runs otherwise. See docs/performance.md for the methodology.

import UIKit
import UltralyticsYOLO

enum Benchmark {
  private static let warmupRuns = 3
  private static let measuredRuns = 15

  /// Writes straight to stderr so the line naming a model survives a crash inside the runtime while loading it.
  private static func log(_ line: String) {
    FileHandle.standardError.write(Data((line + "\n").utf8))
  }

  static func run() {
    DispatchQueue.global(qos: .userInitiated).async {
      let image =
        Bundle.main.url(forResource: "Models/bus", withExtension: "jpg").flatMap {
          CIImage(contentsOf: $0)
        }
        ?? CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 1080, height: 810))
      var rows = [
        "| Model | Task | Compute | Load ms | Pre ms | Inference ms | Post ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
      ]
      for task in appTasks {
        let folder = Bundle.main.url(forResource: task.folder, withExtension: nil)
        let models =
          (folder.flatMap {
            try? FileManager.default.contentsOfDirectory(at: $0, includingPropertiesForKeys: nil)
          } ?? [])
          .filter { ["aimodel", "mlpackage", "mlmodel"].contains($0.pathExtension) }
          .filter { $0.pathExtension != "aimodel" || BasePredictor.isCoreAIAvailable }
          .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for model in models {
          for useGpu in [true, false] {
            let compute = useGpu ? "accelerated" : "cpu"
            log("BENCHMARK loading \(model.lastPathComponent) [\(compute)]")
            guard let (predictor, loadMs) = load(model, task: task.yoloTask, useGpu: useGpu) else {
              rows.append(
                "| \(model.lastPathComponent) | \(task.name) | \(compute) | failed | | | |")
              continue
            }
            var timings = [YOLOResult]()
            for run in 0..<(warmupRuns + measuredRuns) {
              let result = predictor.predictOnImage(image: image)
              if run >= warmupRuns { timings.append(result) }
            }
            func median(_ value: (YOLOResult) -> Double) -> String {
              String(format: "%.2f", timings.map(value).sorted()[timings.count / 2])
            }
            rows.append(
              "| \(model.lastPathComponent) | \(task.name) | \(compute) | \(String(format: "%.0f", loadMs)) | "
                + "\(median { $0.preMs }) | \(median { $0.inferenceMs }) | \(median { $0.postMs }) |"
            )
          }
        }
      }
      log(
        "BENCHMARK results (median of \(measuredRuns) runs after \(warmupRuns) warmup)\n"
          + rows.joined(separator: "\n"))
    }
  }

  private static func load(_ url: URL, task: YOLOTask, useGpu: Bool) -> (BasePredictor, Double)? {
    let start = CACurrentMediaTime()
    let done = DispatchSemaphore(value: 0)
    var predictor: BasePredictor?
    BasePredictor.create(for: task, modelURL: url, useGpu: useGpu) { result in
      if case .failure(let error) = result { log("BENCHMARK load failed: \(error)") }
      predictor = try? result.get()
      done.signal()
    }
    done.wait()  // `create` completes on the main queue; this runs on a background queue
    return predictor.map { ($0, (CACurrentMediaTime() - start) * 1000) }
  }
}
