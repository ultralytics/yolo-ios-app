<!-- Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license -->

# YOLO iOS Inference Performance Toolkit

A small, repeatable workflow for profiling and improving CoreML inference performance of YOLO models in the iOS
SDK. The goal is the fastest, lowest-cost inference: minimize per-frame `model + preprocess + postprocess` time.

## What's here

- **`coreml_profile.py`** — measures a model's `predict()` latency on the Neural Engine vs the CPU, and (with
  `--plan`) reports per-operation compute-device assignment so you can see which ops fall off the ANE.

```bash
# Latency on ANE (CPU_AND_NE) and CPU, 50 runs each:
python3 scripts/perf/coreml_profile.py YOLOiOSApp/Models/Detect/yolo26n.mlpackage

# Add ANE residency / CPU-fallback breakdown:
python3 scripts/perf/coreml_profile.py YOLOiOSApp/Models/Detect/yolo26n.mlpackage --plan
```

Requires macOS with `coremltools>=8`, `numpy`, `pillow`. The `CPU_AND_NE` path is the engine that matters on
iPhone; the Mac ANE gives valid **relative** comparisons between export/op variants. Each compute unit is timed
in its own subprocess so the known macOS-only `MPSGraph` GPU-compiler crash on the `ALL`/`CPU_AND_GPU` paths
can't take down the rest of the run.

## Methodology

Profiling is done on macOS (automated, reproducible) and gives relative deltas. Absolute numbers differ from a
specific iPhone, so changes that depend on engine selection should be confirmed once on-device.

## Findings (yolo26n, Apple-silicon ANE)

Baseline per-model ANE latency (median ms): detect 2.5, seg 3.3, pose 2.9, cls 0.7, obb 7.6, sem 1.9 — roughly
**3× faster than CPU-only** across the board, so keeping work on the ANE is the priority.

**Export / model side is already near-optimal:**

- **8-bit palettization** (`int8=True`) halves model size (4.8 MB → 2.6 MB) with **no latency change** — the ANE
  still computes in fp16, palettized weights are just decompressed at load. Worth adopting for download size, not
  for speed.
- The in-graph end2end decode (`topk` / `gather` / index math in the NMS-free head) shows up as ~20% of the
  static _cost estimate_ on the CPU across all tasks, but removing it entirely only improves wall-clock latency by
  **~4%** — the model is compute-bound on the ANE convolutions. The static cost share is **not** a reliable proxy
  for wall-clock here.
- int8 _activation_ quantization is not pursued: the ANE is fp16-native, so it rarely speeds up ANE latency and
  it risks accuracy.

**Swift postprocessing side — where the win was:**

- The instance-segmentation mask matmul already uses `vDSP_mmul` (optimal).
- Building the per-instance probability maps (`Masks.masks`, a `[[[Float]]]`) by element-wise nested-array
  subscripting cost **~2.35 ms** per segmentation frame for 30 instances at 160×160. Replacing it with bulk row
  copies from the contiguous mask buffer is **~13× faster (~0.18 ms)** with bit-identical output and no API
  change — see `generateCombinedMaskImage` in `Sources/YOLO/Plot.swift`. Since the seg model itself is ~3.3 ms,
  this removed a large chunk of the non-model per-frame cost.

## Reproducing the mask-build benchmark

The Swift micro-benchmark used to validate the segmentation change (current vs bulk-copy) is described in the PR;
it builds with `swiftc -O` and reports ~2.35 ms → ~0.18 ms for 30×160×160.
