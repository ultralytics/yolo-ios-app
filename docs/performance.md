<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

# Real-Time Performance

Canonical record of the on-device and host profiling behind the Ultralytics YOLO iOS SDK's camera and [Core ML](https://developer.apple.com/documentation/coreml) configuration. Each section is a self-contained experiment: the question, the empirical result, and the conclusion. Use it as the starting point and baseline for future performance work.

> [!IMPORTANT]
> **Host benchmarks predict the wrong winner — always confirm on device.** Mac `coremltools` `predict` deltas are useful for quick relative screening, but two separate findings below (end2end head, compute units) **flipped or vanished** when measured on an actual iPhone. Treat host numbers as hypotheses; treat the Xcode Core ML Performance Report and an instrumented on-device build as ground truth.

## 📱 Test Setup

- **Device (ground truth):** iPhone 17 Pro (A19, iOS 26.5.2).
- **Host (relative screening only):** Apple M4 Pro, `coremltools`.
- **Model:** `yolo26n` per task, 640×640 input (1024 for OBB, 224 for classify), int8 Core ML.
- Numbers are EMA-smoothed steady-state. The device thermally settles under sustained use, so figures reflect continuous operation, not a cold burst.

## 📊 Standardized Backend Benchmark

End-to-end `predictOnImage` speeds for the official YOLO26n INT8 Core ML models on the test device
(iPhone 17 Pro, A19, iOS 26.5.2), as **total time** with the preprocess / inference / postprocess split beneath
each value. Annotation drawing is excluded. On iOS, Vision performs input scaling inside the inference request,
so preprocess is reported as 0 and its cost is included in inference.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>`.cpuOnly`<br>(ms)</sup> | Neural Engine<br><sup>`.cpuAndNeuralEngine`<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | ------------------------------------ | --------------------------------------------------------- |
| YOLO26n       | Detect   | 640                         | 9.1<br><sup>0.0 / 9.1 / 0.0</sup>    | **3.8**<br><sup>0.0 / 3.8 / 0.0</sup>                     |
| YOLO26n-seg   | Segment  | 640                         | 12.3<br><sup>0.0 / 12.1 / 0.2</sup>  | **4.8**<br><sup>0.0 / 4.5 / 0.3</sup>                     |
| YOLO26n-sem   | Semantic | 1024<sup>1</sup>            | 21.8<br><sup>0.0 / 21.0 / 0.8</sup>  | **12.1**<br><sup>0.0 / 11.3 / 0.8</sup>                   |
| YOLO26n-depth | Depth    | 640                         | 24.8<br><sup>0.0 / 23.9 / 0.9</sup>  | **5.5**<br><sup>0.0 / 4.7 / 0.9</sup>                     |
| YOLO26n-cls   | Classify | 224                         | 2.2<br><sup>0.0 / 2.2 / 0.0</sup>    | **2.0**<br><sup>0.0 / 2.0 / 0.0</sup>                     |
| YOLO26n-pose  | Pose     | 640                         | 12.0<br><sup>0.0 / 11.9 / 0.0</sup>  | **3.8**<br><sup>0.0 / 3.8 / 0.0</sup>                     |
| YOLO26n-obb   | OBB      | 1024                        | 21.7<br><sup>0.0 / 21.7 / 0.0</sup>  | **7.2**<br><sup>0.0 / 7.2 / 0.0</sup>                     |

- <sup>1</sup> Semantic uses the in-graph ArgMax class-map Core ML export at full resolution
  (ultralytics/ultralytics#24790 + #24799): the argmax runs in the graph and emits a `[1, 1024, 1024]` class map,
  so masks render pixel-sharp and `postProcessSemantic` is a sub-millisecond color sweep.
- **Speed** values are the mean of 15 runs after 3 warmup runs on [bus.jpg](https://ultralytics.com/images/bus.jpg),
  measured through the SDK's per-stage timing (`YOLOResult.preMs`/`inferenceMs`/`postMs`) in profile-mode
  builds (optimized native code).
  <br>From the `example/` directory of the [Flutter plugin Depth PR](https://github.com/ultralytics/yolo-flutter-app/pull/562), reproduce the six established task rows with
  `flutter drive --profile -d <iphone> --driver=test_driver/integration_test.dart --target=integration_test/qnn_benchmark_test.dart --dart-define=RUN_BENCH=true`.
  Reproduce Depth with
  `flutter drive --profile -d <iphone> --driver=test_driver/integration_test.dart --target=integration_test/depth_benchmark_test.dart --dart-define=RUN_DEPTH_BENCH=true`.
  Add `--dart-define=USE_GPU=false` to the Depth command for `.cpuOnly`.
- **These are single-image burst latencies**, not sustained camera frame times: one ~0.9 MP photo through
  `predictOnImage` on a thermally rested device, with no live capture pipeline competing for the ANE. Sustained
  real-time camera operation measures **~11.3 ms/frame** for YOLO26n detect on this same device — see
  [⏱️ What the App's "Inference Time" Actually Measures](#%EF%B8%8F-what-the-apps-inference-time-actually-measures)
  and [📐 High-Resolution Preview, Model-Sized Inference](#-experiment-high-resolution-preview-model-sized-inference)
  for the steady-state pipeline breakdown.
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

The on-screen figure is the **entire** `VNImageRequestHandler.perform` per frame — preprocess + model predict + Swift postprocess — not just the model. The decode runs synchronously inside `perform` (the `VNCoreMLRequest` completion handler). On A19, the previous full-resolution `.photo` capture path measured:

| Stage                                      | Time        | Notes                                                                                       |
| ------------------------------------------ | ----------- | ------------------------------------------------------------------------------------------- |
| Preprocess (camera buffer → 640 letterbox) | dominant    | Cost scales with capture resolution — see below.                                            |
| Model inference                            | ≈7 ms       | In-app; vs ≈1.8 ms in the isolated Performance Report (thermal + live-pipeline contention). |
| Postprocess (Swift decode)                 | ≈0.18 ms    | Raw-pointer reads; negligible.                                                              |
| **Total**                                  | **15.9 ms** | The frame time is the pipeline, not the model head.                                         |

## 📷 Experiment: Camera Capture Resolution

**Q:** How much of the frame is preprocessing, and does capture resolution drive it? **A:** `.photo` delivers full-sensor ~2 MP frames that are downscaled to 640 every frame — the dominant cost. Lowering the preset has **no accuracy impact** (the model always sees a 640 input).

| Camera preset                           | Delivered frame   | Preprocess | Frame time  | FPS    |
| --------------------------------------- | ----------------- | ---------- | ----------- | ------ |
| `.photo` (previous)                     | 1206×1608         | Vision     | 15.9 ms     | 15     |
| `.hd1280x720`                           | 720×1280          | Vision     | 13.3 ms     | 30     |
| **720p preview + 640 output (current)** | 360×640 inference | Vision     | **11.3 ms** | **30** |
| `.vga640x480`                           | 480×640           | Vision     | 13.3 ms     | 24     |
| `.vga640x480`                           | 480×640           | manual     | 8.3 ms      | 25     |

**Shipped: `.hd1280x720` preview with a model-sized data output.** The preview layer still receives the crisp 720p
session, while `AVCaptureVideoDataOutput` asks AVFoundation for only the pixels Vision will consume. The preset is
guarded by `canSetSessionPreset` with a `[requested, .high, .photo]` fallback so startup never regresses on a camera
that can't honor it.

## 📐 Experiment: High-Resolution Preview, Model-Sized Inference

**Q:** Can the app display 720p while sending a smaller frame to Vision? **A:** Yes. On iOS 16+, uncompressed
`AVCaptureVideoDataOutput.videoSettings` supports independent width and height. `AVCaptureVideoPreviewLayer` remains
attached directly to the 720p session; only the inference output is resized.

Optimized Release build, sustained live camera, same scene and model assets:

| Task     | Model input     | Inference buffer | Before  | After       | Change   |
| -------- | --------------- | ---------------- | ------- | ----------- | -------- |
| Detect   | 640 scale-fit   | 360×640          | 13.3 ms | **11.3 ms** | **-15%** |
| Segment  | 640 scale-fit   | 360×640          | 15.8 ms | **13.2 ms** | **-16%** |
| Semantic | 1024 scale-fit  | 576×1024         | 27.4 ms | **22.4 ms** | **-18%** |
| Depth    | 640 scale-fit   | 360×640          | 18.3 ms | **16.5 ms** | **-10%** |
| Classify | 224 center-crop | 224×398          | 8.1 ms  | **7.8 ms**  | **-4%**  |
| Pose     | 640 scale-fit   | 360×640          | 14.4 ms | **11.8 ms** | **-18%** |
| OBB      | 1024 scale-fit  | 576×1024         | 26.8 ms | **24.1 ms** | **-10%** |

The output dimensions come from the loaded model and Vision crop mode, preserve the active camera format's aspect
ratio, and never exceed the session's native buffer. iOS 13–15 keep the previous full-size output because those OS
versions only accept the pixel-format key. A dedicated `AVCapturePhotoOutput` requests the active format's largest
supported still image, runs inference on that exact photo so overlays remain aligned, and then creates the screen-sized
share composite while live inference continues on the smaller buffer. The device test captured and inferred on the
same 2376×4224 photo, then produced the expected 1206×2622-pixel composite.

## 🖼️ Experiment: Preprocessing — Vision vs. Manual vImage

**Q:** How much of the frame is Vision framework overhead vs. the model? **A:** Bypassing Vision with a manual `vImage` letterbox into a reused buffer fed directly to `MLModel.prediction` removes ~5 ms/frame of Vision overhead. (Device, `yolo26n` detect.)

| Path                   | Preprocess             | Inference     | Postprocess | Total    |
| ---------------------- | ---------------------- | ------------- | ----------- | -------- |
| Vision (`.photo`)      | fused in `vis` (≈8 ms) | ≈7 ms (fused) | 0.18 ms     | ≈16 ms   |
| Manual (`.photo`)      | 6.7 ms                 | 7.0 ms        | 0.16 ms     | ≈13.4 ms |
| Manual (`.vga640x480`) | **0.48 ms**            | 7.6 ms        | 0.15 ms     | ≈8.3 ms  |

Manual preprocessing is ~10–15% faster on its own, and stacks with a small capture preset (VGA collapses the letterbox to 0.48 ms → ~8 ms total). It is **not shipped**: it is currently detect-only and its BGRA→model color order needs visual validation. Preserved as an experiment (`git stash`).

## 🎨 Experiment: Segment Mask Painting

**Q:** Can high-resolution instance masks stay sharp without making segment postprocess dominate camera latency?
**A:** Yes. Keep the high-resolution mask path, but make the final color paint pass pointer-based.

Device (`YOLO26n-seg`, live camera, iPhone 17 Pro):

| Segment mask paint path                           | Postprocess |
| ------------------------------------------------- | ----------- |
| Per-detection UIColor lookup + Swift array writes | ≈10 ms      |
| Precomputed color words + pointer ROI paint       | **≈3 ms**   |

The shipped path still scales Float mask logits before thresholding, so mask edges remain high-resolution. The win comes
from removing repeated UIColor component extraction and bounds-checked Swift array writes from the per-pixel ROI sweep.

## 🌡️ Experiment: Depth Map Painting

**Q:** Can a full-resolution depth map be colorized every camera frame without dominating latency? **A:** Yes. Apply the
same Accelerate pattern as semantic segmentation: bulk-copy contiguous tensor rows, use vDSP/vForce for min/max and
log normalization, then paint through Planar8 lookup tables and `vImageConvert_Planar8toARGB8888`.

Device (`YOLO26n-depth`, 640×640 int8 Core ML, live 720p camera, Debug build, 360×640 map after letterbox crop):

| Depth color path                               | Postprocess |
| ---------------------------------------------- | ----------- |
| Per-pixel Swift log + color-stop interpolation | ≈70 ms      |
| Accelerate/vImage vectorized paint             | **≈2.1 ms** |

The vectorized path preserves the public metric-depth array and the same near-to-far color gradient; it only replaces
the scalar rendering sweep.

Single-image burst latency across every official Depth size on the same iPhone 17 Pro (15 runs after 3 warmups,
`bus.jpg`, profile-mode Flutter harness, 480×640 typed metric map):

| Depth model | CPU inference | CPU post | CPU total | Neural Engine inference | NE post | NE total     |
| ----------- | ------------- | -------- | --------- | ----------------------- | ------- | ------------ |
| YOLO26n     | 23.90 ms      | 0.85 ms  | 24.75 ms  | 4.67 ms                 | 0.87 ms | **5.54 ms**  |
| YOLO26s     | 33.67 ms      | 0.90 ms  | 34.57 ms  | 6.15 ms                 | 0.85 ms | **7.01 ms**  |
| YOLO26m     | 55.27 ms      | 0.93 ms  | 56.21 ms  | 9.64 ms                 | 0.89 ms | **10.54 ms** |
| YOLO26l     | 67.32 ms      | 0.93 ms  | 68.25 ms  | 10.87 ms                | 0.94 ms | **11.80 ms** |
| YOLO26x     | 116.77 ms     | 0.94 ms  | 117.71 ms | 19.38 ms                | 0.93 ms | **20.30 ms** |

Vision performs scaling inside the request, so preprocessing is reported as 0 and included in inference. These burst
numbers are distinct from the sustained 16.5 ms/frame YOLO26n Depth camera result above.

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

The same applies to the **Ultralytics Python package on a Mac**: its CoreML backend defaulted to `ComputeUnit.ALL`, so `YOLO("model.mlpackage").predict()` crashed on macOS hosts. Fixed in [ultralytics#24885](https://github.com/ultralytics/ultralytics/pull/24885) — it now loads `CPU_AND_NE` (ANE, ~3× faster; `CPU_ONLY` fallback on macOS <13), so host inference and `val()` run on the Neural Engine out of the box.

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

## 📦 Experiment: Core ML Export Variants

Fresh `yolo26n` exports from the sibling Ultralytics checkout were installed into the optimized app and measured with
the model-sized inference buffer. Reference-image results use `bus.jpg` at the existing 0.25 confidence threshold.

| Export                       | Device total | Package | Reference result              | Decision                                          |
| ---------------------------- | ------------ | ------- | ----------------------------- | ------------------------------------------------- |
| INT8 end2end, 640 (current)  | 11.5–12.0 ms | 2.6 MB  | 5 boxes; top confidence 0.925 | **Keep**                                          |
| FP16 end2end, 640            | 11.4–11.6 ms | 4.8 MB  | 5 boxes; top confidence 0.922 | Within thermal/run noise; 2× size                 |
| INT8 end2end, 480            | 9.9–10.3 ms  | 2.6 MB  | 6 boxes; top confidence 0.872 | Faster, but visibly weaker scores/extra detection |
| INT8 legacy raw, 640         | 12.8–13.0 ms | 2.6 MB  | 5 boxes                       | Rejected: Swift NMS adds ≈2.4 ms                  |
| INT8 Core ML/Vision NMS, 640 | 11.2–11.7 ms | 2.6 MB  | 5 boxes                       | No repeatable win over end2end                    |

The 480 export is the only material inference-side speed lever, but it changes model quality and therefore is not a
drop-in optimization. FP16 and Vision NMS do not justify replacing the current release assets. The existing INT8
end2end export remains the best size/speed/quality default.

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

`.hd1280x720` preview · model-sized inference output · high-resolution photo capture · `.cpuAndNeuralEngine` · INT8
end2end YOLO26 models · Vision preprocessing · optimized high-resolution segment/depth painting · default
`minimum_deployment_target`.

On A19, frame time is dominated by model inference plus Vision's fused scaling. The isolated Performance Report's
~1.8 ms model time is not achievable inside a sustained live camera pipeline.

## 🔓 Open Levers (Untested / Not Shipped)

- ~~**In-graph ArgMax for semantic models.**~~ **Shipped** (ultralytics/ultralytics#24790 + #24799 + this SDK's
  `SemanticSegmenter` class-map support): semantic Core ML exports now embed the ArgMax and return a full-resolution
  `[1, H, W]` class map, replacing the CPU argmax decode with a sub-millisecond color sweep and making masks
  pixel-sharp. The same recipe is QNN/Core ML-only: the LiteRT GPU delegate cannot compile `ARG_MAX` (whole-graph
  CPU fallback measured 3.6× slower than GPU logits), so Android LiteRT keeps consumer-side argmax.

- **Cross-platform decode parity (context, not a lever).** The Flutter Android predictors previously spent
  ~12 ms/frame on detect decode (tensor reshape copies + JNI marshaling); rewriting them to direct flat reads —
  the approach this SDK has always used via raw pointers (~0.18 ms) — took Android detect from 23.3 to 13.1 ms
  end-to-end. Validation that the raw-pointer decode pattern here is the right baseline and should be preserved in
  any future refactor.

- ~~**Lower model input resolution.**~~ Tested at `imgsz=480`: ≈2 ms faster, but lower reference confidences and an
  extra detection. Not shipped because it is an accuracy/quality setting, not a free optimization.
- **Manual vImage preprocessing** — the historical full-resolution experiment saved ~5 ms/frame, but the shipped
  model-sized capture output now removes much of that resize work while retaining Vision. Any remaining benefit needs
  a fresh all-task comparison plus BGRA color-order validation before replacing the current path.
- **Higher camera frame rate** — raise `activeVideoMinFrameDuration` toward 60 fps where the format and lighting allow, now that inference has headroom.
- **Frame skipping** — process every Nth frame to cut sustained power/thermal in always-on use.
