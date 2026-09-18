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
  private static let rounds = 3
  private static let runsPerRound = 15

  /// Writes straight to stderr so the line naming a model survives a crash inside the runtime while loading it.
  private static func log(_ line: String) {
    FileHandle.standardError.write(Data((line + "\n").utf8))
  }

  static func run() {
    DispatchQueue.global(qos: .userInitiated).async {
      guard
        let image = Bundle.main.url(forResource: "Models/bus", withExtension: "jpg").flatMap({
          CIImage(contentsOf: $0)
        })
      else {
        log("BENCHMARK needs Models/bus.jpg: run scripts/download-models.sh --coreai")
        return
      }
      var rows = [
        "| Model | Task | Compute | Load ms | Pre ms | Inference ms | Post ms | Result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
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

        // Load every model of the task first, then time them in interleaved rounds so thermal drift and run order
        // affect all backends equally.
        // Only the timings are kept per run: results hold images and masks, and retaining them all exhausts memory.
        var entries = [
          (
            name: String, loadMs: Double, predictor: BasePredictor, found: String,
            timings: [[Double]]
          )
        ]()
        for model in models {
          for useGpu in [true, false] {
            let name =
              "\(model.lastPathComponent) | \(task.name) | \(useGpu ? "accelerated" : "cpu")"
            log("BENCHMARK loading \(name)")
            guard let (predictor, loadMs) = load(model, task: task.yoloTask, useGpu: useGpu) else {
              rows.append("| \(name) | failed | | | | |")
              continue
            }
            var found = ""
            for _ in 0..<warmupRuns { found = summary(predictor.predictOnImage(image: image)) }
            log("BENCHMARK warmed up \(name): \(found)")
            entries.append((name, loadMs, predictor, found, []))
          }
        }
        for _ in 0..<rounds {
          for index in entries.indices {
            for _ in 0..<runsPerRound {
              autoreleasepool {
                let result = entries[index].predictor.predictOnImage(image: image)
                entries[index].timings.append([result.preMs, result.inferenceMs, result.postMs])
              }
            }
          }
        }
        for entry in entries {
          func median(_ stage: Int) -> String {
            String(
              format: "%.2f", entry.timings.map { $0[stage] }.sorted()[entry.timings.count / 2])
          }
          rows.append(
            "| \(entry.name) | \(String(format: "%.0f", entry.loadMs)) | \(median(0)) | "
              + "\(median(1)) | \(median(2)) | \(entry.found) |")
        }
      }
      log(
        "BENCHMARK results (median of \(rounds) interleaved rounds of \(runsPerRound) runs after "
          + "\(warmupRuns) warmup, bus.jpg)\n" + rows.joined(separator: "\n"))
    }
  }

  /// What the model found, so a backend that runs fast but decodes nothing is visible in the table.
  private static func summary(_ result: YOLOResult) -> String {
    if let probs = result.probs { return "\(probs.top1) \(String(format: "%.3f", probs.top1Conf))" }
    let confidences = result.obb.isEmpty ? result.boxes.map(\.conf) : result.obb.map(\.confidence)
    if let mask = result.semanticMask { return "\(Set(mask.classMap).count) classes" }
    if let depth = result.depthMap {
      return String(format: "depth %.2f-%.2f", depth.minDepth, depth.maxDepth)
    }
    return "\(confidences.count) @ "
      + confidences.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: " ")
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
