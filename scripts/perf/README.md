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

**Swift postprocessing side — where the wins are.** The decode loops for detect/pose/OBB (≤300 detections, raw
pointer access) are already optimal, and the segmentation mask matmul already uses `vDSP_mmul`. The wins were in
array marshaling around them. Each was validated with a `swiftc -O` micro-benchmark and locked with a test:

| Task     | Change                                                                                                                                                                                                                                | Before → After                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| segment  | Per-instance probability maps (`Masks.masks`, `[[[Float]]]`) built by element-wise nested-array subscripting → bulk row copies from the contiguous buffer. Bit-identical, no API change. `generateCombinedMaskImage` in `Plot.swift`. | **2.35 ms → 0.18 ms** (30×160×160)                   |
| segment  | Per-detection mask coefficients held in a heavyweight `MLMultiArray` → plain `[Float]`. Also removes a force-`try` and simplifies the code. `Segmenter.swift`.                                                                        | **0.144 ms → 0.026 ms** (300 dets)                   |
| semantic | Per-pixel class argmax over NCHW logits (each class read `H*W` apart, cache-thrashing) → cache-friendly class-major pass. `postProcessSemantic` in `SemanticSegmenter.swift`.                                                         | **1.11 ms → 0.80 ms** (19×320×320); smaller at 80×80 |

**General lesson for this repo:** the model runs on the ANE and is already fast; the Swift-side hotspots are
nested-array / heavyweight-object marshaling in per-frame postprocessing, not the numeric decode loops.

## Reproducing the micro-benchmarks

Each change was validated by a standalone `swiftc -O` benchmark comparing the old and new implementation on
representative shapes and asserting identical output. The numbers above are medians on Apple silicon.
