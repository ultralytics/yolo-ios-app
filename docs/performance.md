<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

# Real-Time Performance

Canonical record of the on-device and host profiling behind the Ultralytics YOLO iOS SDK's camera and [Core ML](https://developer.apple.com/documentation/coreml) configuration. Each section is a self-contained experiment: the question, the empirical result, and the conclusion. Use it as the starting point and baseline for future performance work.

> [!IMPORTANT]
> **Host benchmarks predict the wrong winner — always confirm on device.** Mac `coremltools` `predict` deltas are useful for quick relative screening, but two separate findings below (end2end head, compute units) **flipped or vanished** when measured on an actual iPhone. Treat host numbers as hypotheses; treat the Xcode Core ML Performance Report and an instrumented on-device build as ground truth.

## 📱 Test Setup

- **Device (ground truth):** iPhone 17 Pro (A19, iOS 26.5).
- **Host (relative screening only):** Apple M4 Pro, `coremltools`.
- **Model:** `yolo26n` per task, 640×640 input (1024 for OBB, 224 for classify), int8 Core ML.
- Numbers are EMA-smoothed steady-state. The device thermally settles under sustained use, so figures reflect continuous operation, not a cold burst.

## 📊 Standardized Backend Benchmark

End-to-end `predictOnImage` speeds for the official YOLO26n INT8 Core ML models on the test device
(iPhone 17 Pro, A19, iOS 26.5), as **total time** with the preprocess / inference / postprocess split beneath
each value. Annotation drawing is excluded. On iOS, Vision performs input scaling inside the inference request,
so preprocess is reported as 0 and its cost is included in inference.

| Model        | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>`.cpuOnly`<br>(ms)</sup> | Neural Engine<br><sup>`.cpuAndNeuralEngine`<br>(ms)</sup> |
| ------------ | -------- | --------------------------- | ------------------------------------ | --------------------------------------------------------- |
| YOLO26n      | Detect   | 640                         | 9.2<br><sup>0.0 / 9.1 / 0.0</sup>    | **3.8**<br><sup>0.0 / 3.7 / 0.0</sup>                     |
| YOLO26n-seg  | Segment  | 640                         | 21.7<br><sup>0.0 / 12.0 / 9.8</sup>  | **14.1**<br><sup>0.0 / 4.5 / 9.6</sup>                    |
| YOLO26n-sem  | Semantic | 1024                        | 14.6<br><sup>0.0 / 7.4 / 7.3</sup>   | **10.3**<br><sup>0.0 / 3.3 / 7.0</sup>                    |
| YOLO26n-cls  | Classify | 224                         | 2.4<br><sup>0.0 / 2.4 / 0.0</sup>    | **2.0**<br><sup>0.0 / 1.9 / 0.0</sup>                     |
| YOLO26n-pose | Pose     | 640                         | 12.1<br><sup>0.0 / 12.0 / 0.1</sup>  | **3.9**<br><sup>0.0 / 3.9 / 0.1</sup>                     |
| YOLO26n-obb  | OBB      | 1024                        | 22.3<br><sup>0.0 / 22.3 / 0.0</sup>  | **7.2**<br><sup>0.0 / 7.2 / 0.0</sup>                     |

- **Speed** values are the mean of 15 runs after 3 warmup runs on [bus.jpg](https://ultralytics.com/images/bus.jpg),
  measured through the SDK's per-stage timing (`YOLOResult.preMs`/`inferenceMs`/`postMs`).
  <br>Reproduce with the Flutter plugin's harness:
  `flutter test integration_test/qnn_benchmark_test.dart -d <iphone> --dart-define=RUN_BENCH=true`
- **These are single-image burst latencies**, not sustained camera frame times: one ~0.9 MP photo through
  `predictOnImage` on a thermally rested device, with no live capture pipeline competing for the ANE. Sustained
  real-time camera operation measures **~16 ms/frame** for YOLO26n detect on this same device — see
  [⏱️ What the App's "Inference Time" Actually Measures](#%EF%B8%8F-what-the-apps-inference-time-actually-measures)
  for the steady-state pipeline breakdown (full-sensor letterboxing dominates, and in-app inference settles at ~7 ms
  vs ~1.8 ms isolated).
- The matching Snapdragon CPU/GPU/NPU table lives in the
  [Flutter plugin performance guide](https://github.com/ultralytics/yolo-flutter-app/blob/main/doc/performance.md).

## 🔬 Methodology (How to Reproduce)

| Tool                                                                                              | What it measures                                                         | Notes                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coremltools` `MLModel.predict` per `MLComputeUnit`, subprocess-isolated, interleaved round-robin | host latency, relative deltas                                            | Interleaving cancels thermal drift. `ALL` / `CPU_AND_GPU` `predict` **crash** the Mac host (MPSGraph compiler bug); only `CPU_ONLY` and `CPU_AND_NE` are usable host-side.                                                                                       |
| `MLComputePlan.load_from_path` on a compiled `.mlmodelc`                                          | per-op preferred device + estimated cost (ANE/GPU/CPU residency)         | Static plan; the CPU **cost share** is **not** a wall-clock proxy.                                                                                                                                                                                               |
| Xcode **Core ML Performance Report** (`.mlpackage` → Performance → device → All)                  | per-layer on-device compute-unit placement + prediction latency          | Gold standard for absolute device numbers; exports `*.mlperf/report.json`.                                                                                                                                                                                       |
| Instrumented app build (experiment branch; these hooks are not in the shipped SDK)                | per-frame `preprocess / inference / postprocess` split, EMA + raw jitter | Tap the FPS label to A/B Vision↔manual; env `YOLO_PREPROCESS`, `YOLO_COMPUTE_UNITS`, `YOLO_CAMERA_PRESET`; `[perf]` to stdout (capture via `xcrun devicectl device process launch --console`) and `os_log` (`subsystem com.ultralytics.yolo`, category `YOLO`). |

## ⏱️ What the App's "Inference Time" Actually Measures

The on-screen figure is the **entire** `VNImageRequestHandler.perform` per frame — preprocess + model predict + Swift postprocess — not just the model. The decode runs synchronously inside `perform` (the `VNCoreMLRequest` completion handler). On A19, `yolo26n` detect at `.photo` capture:

| Stage                                      | Time       | Notes                                                                                       |
| ------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------- |
| Preprocess (camera buffer → 640 letterbox) | dominant   | Cost scales with capture resolution — see below.                                            |
| Model inference                            | ≈7 ms      | In-app; vs ≈1.8 ms in the isolated Performance Report (thermal + live-pipeline contention). |
| Postprocess (Swift decode)                 | ≈0.18 ms   | Raw-pointer reads; negligible.                                                              |
| **Total**                                  | **≈16 ms** | The "16 ms" is the pipeline, not the model head.                                            |

## 📷 Experiment: Camera Capture Resolution

**Q:** How much of the frame is preprocessing, and does capture resolution drive it? **A:** `.photo` delivers full-sensor ~2 MP frames that are downscaled to 640 every frame — the dominant cost. Lowering the preset has **no accuracy impact** (the model always sees a 640 input).

| Camera preset               | Delivered frame | Preprocess | Frame time  | FPS    |
| --------------------------- | --------------- | ---------- | ----------- | ------ |
| `.photo` (previous)         | 1206×1608       | Vision     | 15.9 ms     | 15     |
| **`.hd1280x720` (current)** | 720×1280        | Vision     | **13.3 ms** | **30** |
| `.vga640x480`               | 480×640         | Vision     | 13.3 ms     | 24     |
| `.vga640x480`               | 480×640         | manual     | 8.3 ms      | 25     |

**Shipped: `.hd1280x720`** — cuts pipeline frame time vs `.photo` (15.9 → 13.3 ms) and doubles sustained FPS while keeping a crisp 16:9 preview. `.vga640x480` (4:3, same FOV as `.photo`) lands on 640 with pad-only/no-downscale, the cheapest preprocess. The preset is guarded by `canSetSessionPreset` with a `[requested, .high, .photo]` fallback so startup never regresses on a camera that can't honor it.

## 🖼️ Experiment: Preprocessing — Vision vs. Manual vImage

**Q:** How much of the frame is Vision framework overhead vs. the model? **A:** Bypassing Vision with a manual `vImage` letterbox into a reused buffer fed directly to `MLModel.prediction` removes ~5 ms/frame of Vision overhead. (Device, `yolo26n` detect.)

| Path                   | Preprocess             | Inference     | Postprocess | Total    |
| ---------------------- | ---------------------- | ------------- | ----------- | -------- |
| Vision (`.photo`)      | fused in `vis` (≈8 ms) | ≈7 ms (fused) | 0.18 ms     | ≈16 ms   |
| Manual (`.photo`)      | 6.7 ms                 | 7.0 ms        | 0.16 ms     | ≈13.4 ms |
| Manual (`.vga640x480`) | **0.48 ms**            | 7.6 ms        | 0.15 ms     | ≈8.3 ms  |

Manual preprocessing is ~10–15% faster on its own, and stacks with a small capture preset (VGA collapses the letterbox to 0.48 ms → ~8 ms total). It is **not shipped**: it is currently detect-only and its BGRA→model color order needs visual validation. Preserved as an experiment (`git stash`).

## 🧠 Experiment: Core ML Compute Units (CPU / GPU / ANE)

**Q:** Should inference use `.cpuAndNeuralEngine` or `.all` (adds GPU)? **A:** `.all` is no faster and slightly jitterier in a live camera app, where the GPU is busy compositing the preview/overlays. The model is ~7 ms in-app under **both**.

Device, `yolo26n` detect, 4-cell matrix (raw = un-smoothed per-frame ms):

| Preprocess | Compute units         | Model `inf` | Total   | Raw min/p90/max    |
| ---------- | --------------------- | ----------- | ------- | ------------------ |
| Vision     | `.cpuAndNeuralEngine` | fused       | 16.0 ms | 13.2/17.3/17.8     |
| Vision     | `.all`                | fused       | 16.4 ms | 12.7/17.6/**18.8** |
| Manual     | `.cpuAndNeuralEngine` | 6.95 ms     | 14.0 ms | 10.7/15.0/18.5     |
| Manual     | `.all`                | 6.80 ms     | 13.0 ms | 11.0/13.9/16.8     |

Host (`coremltools`, `yolo26n`): `CPU_AND_NE` **2.6 ms** vs `CPU_ONLY` **8.6 ms** — the ANE is ~3× faster than CPU. (GPU-only is not measurable host-side; `ALL`/`CPU_AND_GPU` crash the Mac.)

**Shipped: `.cpuAndNeuralEngine`** — keeps the conv backbone on the ANE and avoids GPU contention. Do not switch to `.all`.

## 🎯 Experiment: YOLO26 End2end Head vs. Legacy Head + NMS

**Q:** The YOLO26 in-graph end2end decode (top-k/gather, no NMS) puts some ops on the CPU — is it slower than a legacy head + Core ML/Vision NMS? **A:** On host it looks slower; **on device it is as fast or faster.** Keep end2end.

Host (`coremltools` NE median, interleaved, `yolo26n`):

| Variant (`yolo26n`)                          | NE latency | ANE op-share |
| -------------------------------------------- | ---------- | ------------ |
| end2end (`end2end=True, nms=False`, shipped) | 2.60 ms    | 92.9%        |
| legacy raw (`end2end=False, nms=False`)      | 2.26 ms    | 99.3%        |
| legacy + Core ML NMS (`nms=True`)            | 2.37 ms    | (pipeline)   |

Host end2end penalty vs legacy+NMS: **n +9.4%, s +6.4%, m +3.9%** (fixed ~0.25 ms CPU decode cost, so a larger % on smaller models). Compute plan: end2end adds ~19 CPU ops (top-k/gather/decode), dropping ANE op-share 99.3%→92.9%.

Device (Xcode Performance Report, A19, `yolo26n`):

| Variant      | Median  | Min     | CPU ops | ANE ops |
| ------------ | ------- | ------- | ------- | ------- |
| end2end      | 1.81 ms | 1.52 ms | 21      | 276     |
| legacy + NMS | 1.90 ms | 1.76 ms | 2       | 282     |

**On device the sign flips: end2end is faster** — the A19 ANE runs the in-graph top-k cheaper than a separate Vision NMS stage. The 21 CPU ops are real but cheap. Re-exporting `end2end=False` for speed is a no-op-to-loss on device, and would force Swift-side NMS for OBB/pose/seg. The end2end export stays the default.

## 🆚 Experiment: YOLO26 vs. YOLO11 Backbone

**Q:** Is the YOLO26 backbone slower than YOLO11 on Core ML? **A:** No — they're equal. Host (`coremltools` NE, raw heads): `yolo26n` 2.26 ms ≈ `yolo11n` 2.25 ms (and `yolo26n` is smaller, 2.71 vs 2.89 MB). Any "YOLO26 is slower" impression was the end2end head, not the backbone.

## ⚖️ Experiment: Quantization & Deployment Target

| Setting                               | Effect                                                                                               |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| int8 (8-bit palettization, shipped)   | ~½ model size, **no latency change** (ANE computes fp16; weights decompress at load). Size win only. |
| int8 activation quantization          | No benefit (the ANE is fp16-native).                                                                 |
| `minimum_deployment_target` iOS17/18+ | **Regresses** detect latency (adds casts / CPU ops). Leave unset; let `coremltools` choose.          |

## 🎞️ Frame Rate Is Camera-Bound, Not Inference-Bound

`FPS` is the rate the camera delivers and the pipeline processes frames — **not** `1000 / inference_ms`. At ≤13 ms/frame the pipeline finishes well inside the camera's frame interval and idles until the next frame. Proof: manual+VGA (8.3 ms) ran at a _lower_ FPS than Vision+720p (13.3 ms). The cap is set by the camera format (~30 fps default), reduced in dim light (auto-exposure lengthens frame duration) and under thermal load — and the SDK does not raise `activeVideoMinFrameDuration`. Faster inference therefore buys latency and power/thermal headroom, not FPS, unless the camera frame-rate cap is raised.

## 📐 Aspect-Ratio Robustness (16:9 ↔ 4:3)

Capture presets differ in aspect (`.photo`/`.vga640x480` are 4:3, `.hd1280x720` is 16:9), so letterbox bars fall on different axes. The pipeline is aspect-agnostic: `letterboxTransform` derives `gain = min(640/W, 640/H)` with independent centered `padX`/`padY` from the **live** frame size; `inputRect` inverts it; and on-screen overlays use aspect-**fill** mapping consistent with the preview's `.resizeAspectFill` gravity. Verified by `Tests/YOLOTests/LetterboxTests.swift` (round-trips both aspects in both orientations).

## ✅ Shipped Configuration

`.hd1280x720` capture · `.cpuAndNeuralEngine` · int8 end2end YOLO26 models · Vision preprocessing · default `minimum_deployment_target`.

On A19, frame time is dominated by model inference (~7 ms, thermally bound under sustained live use) plus preprocessing. The isolated Performance Report's ~1.8 ms model time is not achievable inside a live camera pipeline.

## 🔓 Open Levers (Untested / Not Shipped)

- **Lower model input resolution** — re-export at `imgsz=480` (0.56× the pixels of 640) to cut the ~7 ms model time; the largest remaining lever.
- **Manual vImage preprocessing** — ~5 ms/frame win; needs BGRA color-order validation and per-task (OBB/pose/seg) decode support before shipping.
- **Higher camera frame rate** — raise `activeVideoMinFrameDuration` toward 60 fps where the format and lighting allow, now that inference has headroom.
- **Frame skipping** — process every Nth frame to cut sustained power/thermal in always-on use.
