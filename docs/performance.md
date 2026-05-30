<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

# Real-Time Performance

This document records on-device performance profiling of the Ultralytics YOLO iOS SDK and the rationale behind its camera and [Core ML](https://developer.apple.com/documentation/coreml) configuration. Numbers are intended as relative guidance; absolute values vary by device, thermal state, and lighting.

## Methodology

Measured on a physical **iPhone 17 Pro (A19, iOS 26.5)** running `yolo26n` detection at a 640×640 model input. Each camera frame's wall-clock time is split into three stages:

- **Preprocess** — convert the camera pixel buffer to the model's 640×640 input (scale + letterbox + color).
- **Inference** — Core ML `predict` on the Apple Neural Engine.
- **Postprocess** — Swift-side decode of the model output into results.

Values are EMA-smoothed steady-state. Under sustained real-time use the device thermally settles, so figures reflect typical continuous operation rather than a brief cold burst.

## Camera capture resolution

The camera capture preset sets how much image data is scaled to the 640×640 model input every frame. The SDK previously captured at `.photo` (full sensor), which delivers ~2 MP frames that must be downscaled on every frame — the dominant preprocessing cost. Capturing at `.hd1280x720` roughly halves the per-frame data and doubles sustained throughput, with **no change to detection accuracy** (the model always receives a 640 input; a lower-resolution capture simply means less downscaling).

| Camera preset            | Delivered frame | Preprocess | Frame time | FPS |
| ------------------------ | --------------- | ---------- | ---------- | --- |
| `.photo` (previous)      | 1206×1608       | Vision     | 15.9 ms    | 15  |
| **`.hd1280x720` (current)** | 720×1280     | Vision     | **13.3 ms** | **30** |
| `.vga640x480`            | 480×640         | Vision     | 13.3 ms    | 24  |
| `.vga640x480`            | 480×640         | manual†    | 8.3 ms     | 25  |

The SDK ships with **`.hd1280x720`**: it halves frame time versus `.photo` and doubles the sustained frame rate while keeping a crisp preview.

† A manual `vImage` letterbox that bypasses Vision (feeding the model a pre-letterboxed buffer directly) removes Vision's per-frame framework overhead and, at VGA, collapses preprocessing to ~0.5 ms. It is shown as an experimental ceiling and is **not** the shipped path.

### Frame rate is camera-bound, not inference-bound

`FPS` is the rate at which the camera delivers and the pipeline processes frames — **not** `1000 / inference_ms`. At 13 ms/frame the pipeline finishes well inside the camera's frame interval and idles until the next frame. The capture rate is capped by the camera format (≈30 fps by default), is reduced in dim light (auto-exposure lengthens frame duration), and drops under thermal load. Faster inference therefore buys lower latency and lower power/heat rather than a higher frame rate, unless the camera's frame-rate cap is also raised.

### Aspect-ratio robustness (16:9 ↔ 4:3)

`.photo` is 4:3 and `.hd1280x720` is 16:9, so the letterbox bars fall on different axes (left/right for tall frames, top/bottom for wide frames). The preprocessing is aspect-agnostic: the letterbox transform computes `gain = min(640/W, 640/H)` with independent, centered `padX`/`padY` from the **live** frame dimensions, and the inverse mapping (`inputRect`) restores detections to the full frame. On-screen, bounding boxes use aspect-**fill** mapping that matches the preview's `.resizeAspectFill` gravity, so overlays stay aligned across formats. This is covered by `LetterboxTests`, which round-trips both aspect ratios in both orientations.

## Core ML compute units

Inference uses `MLComputeUnits.cpuAndNeuralEngine`. An on-device A/B versus `.all` showed `.all` is **no faster** (model inference ≈7 ms either way) and slightly increases frame-time jitter in the camera pipeline, where the GPU is already compositing the preview and overlays. `.cpuAndNeuralEngine` keeps the convolutional backbone on the Apple Neural Engine and avoids that contention.

## YOLO26 NMS-free (end2end) head

YOLO26 detection models export with an in-graph end2end head (top-k decode, no NMS). On-device this is as fast as — or faster than — a legacy head plus a Core ML/Vision NMS pipeline, even though a few decode ops fall to the CPU. The end2end export is the SDK default.

## Postprocessing

Swift-side decode is ≈0.2 ms/frame (raw-pointer tensor reads; segmentation mask assembly uses `vDSP` and bulk row copies). It is not a meaningful share of frame time.

## Summary

For `yolo26n` detection on A19, frame time is dominated by model inference (≈7 ms, thermally bound under sustained use) and image preprocessing. The shipped configuration — `.hd1280x720` capture, `.cpuAndNeuralEngine`, end2end models — reflects on-device measurement rather than isolated benchmarks, where an idle, cool Neural Engine can report substantially lower per-inference times than are achievable inside a live camera pipeline.
