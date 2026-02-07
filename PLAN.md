# YOLO iOS SDK — 2026 Modernization Plan

## Core Principles (CRITICAL)

Every agent — implementer or reviewer — MUST follow these principles. Violations should be flagged immediately.

1. **Minimal**: Simplest solution that works — no over-engineering
2. **Replace > Add**: Modify existing code over adding new
3. **Delete ruthlessly**: Remove unused code completely. No `// removed`, no `_unused` vars, no dead paths
4. **Check existing**: Search ALL files in the repo before creating new utilities, types, helpers, or constants
5. **No duplication**: If logic exists somewhere, reuse it. If you extract shared code, delete the original
6. **Cross-app awareness**: If a pattern exists in one app or module, check if it can be reused or adapted elsewhere
7. **Code cleanup**: Every time you add or replace code, you MUST clean up what you replaced. Dead code accumulates quickly and creates confusion, maintenance burden, and technical debt

These principles override any impulse to "add a helper," "create an abstraction," or "add for future use." Three similar lines of code is better than a premature abstraction.

---

## Agent Team Instructions

When implementing this plan, Claude should spawn a team of agents organized as follows.

### Team structure

```
team-lead (coordinator)
├── engine-impl       (general-purpose agent — implements YOLOCore)
├── ui-impl           (general-purpose agent — implements YOLOUI + SwiftUI app)
└── reviewer          (general-purpose agent — reviews all changes)
```

### Spawning instructions

```
Use TeamCreate to create team "yolo-modernize".

Create tasks for each phase using TaskCreate, then spawn teammates using Task tool:

1. "engine-impl" — subagent_type: general-purpose, mode: plan
   Implements Phases 1, 2, 3 (YOLOCore extraction, YOLO26 NMS-free, YOLOSession)

2. "ui-impl" — subagent_type: general-purpose, mode: plan
   Implements Phases 4, 5, 6 (YOLOUI SwiftUI views, example app, App Store app)
   Blocked by: engine-impl completing Phase 1

3. "reviewer" — subagent_type: general-purpose, mode: plan
   Reviews every PR/changeset from the other 2 agents before merge
   Runs Phase 7 (concurrency audit) continuously
```

### Instructions for ALL agents

```
You are working on the YOLO iOS SDK modernization. Read PLAN.md at repo root for full context.

## Core Principles (MUST FOLLOW)
1. Minimal: Simplest solution that works — no over-engineering
2. Replace > Add: Modify existing code over adding new
3. Delete ruthlessly: Remove unused code completely
4. Check existing: Search ALL files before creating utilities/types/constants
5. No duplication: Reuse existing logic, delete originals after extraction
6. Cross-app awareness: Check if patterns exist elsewhere before creating new ones
7. Code cleanup: Every add/replace MUST clean up what it replaced

## Before writing ANY code:
- Read the existing file first
- Search for similar patterns in the codebase (Grep/Glob)
- Confirm the code you're replacing is actually unused after your change
- Delete dead code in the same commit — never leave it for later

## After writing code:
- Grep for any remaining references to code you replaced
- Verify no duplicate logic exists
- Ensure imports are minimal (no unused imports)
- Run swift build to verify compilation
```

### Additional instructions for REVIEWER agent

```
You review all changes from engine-impl and ui-impl.

For every changeset, check against Core Principles:
1. MINIMAL — Is this the simplest way? Could it be done with less code? Are there unnecessary
   abstractions, helpers, or indirection layers?
2. REPLACE > ADD — Did the agent modify existing files, or did they create new ones unnecessarily?
   New files are only acceptable when the plan explicitly calls for them.
3. DELETE RUTHLESSLY — Is there any dead code left behind? Old imports? Commented-out code?
   Unused parameters? Empty protocol conformances?
4. CHECK EXISTING — Did the agent duplicate something that already exists in the codebase?
   Search for similar function names, type names, and patterns.
5. NO DUPLICATION — Is the same logic now in two places? If shared code was extracted,
   was the original deleted?
6. CROSS-APP AWARENESS — Does this change affect YOLOiOSApp, ExampleApps, or tests?
   Were those updated too?
7. CODE CLEANUP — For every line added, was the replaced line removed? Check git diff
   for any "+ new code" without corresponding "- old code".

Also verify:
- No @unchecked Sendable
- No DispatchQueue.main.async (use @MainActor)
- No completion handler callbacks for new code (use async/await)
- No UIKit views anywhere — SwiftUI only (UIViewRepresentable only for AVCaptureVideoPreviewLayer)
- No UIKit imports in YOLOCore (except AVFoundation for camera)
- Swift 6 strict concurrency compliance
- YOLO26 (not YOLO11) in all new code and docs

REJECT changes that violate any principle. Provide specific line numbers and fixes.
```

---

## Current State

The `YOLO` Swift Package started as a YOLO11 detection-only demo and grew into a multi-task framework. The architecture hasn't kept up. Models are YOLO11 (requires NMS), while current-gen YOLO26 is NMS-free.

### Current files (`Sources/YOLO/` — 19 files)

| File | Lines | Role |
|------|-------|------|
| `YOLO.swift` | 235 | Public API — model loading + single-image inference |
| `YOLOView.swift` | 1413 | God-view — camera, UI, rendering, ALL tasks |
| `YOLOCamera.swift` | 83 | `UIViewRepresentable` wrapper around `YOLOView` |
| `BasePredictor.swift` | 339 | CoreML model loading + inference |
| `Predictor.swift` | 92 | Protocol definitions |
| `ObjectDetector.swift` | — | Detection post-processing |
| `Classifier.swift` | — | Classification post-processing |
| `Segmenter.swift` | — | Segmentation post-processing |
| `PoseEstimator.swift` | — | Pose estimation post-processing |
| `ObbDetector.swift` | — | OBB post-processing |
| `YOLOResult.swift` | 253 | Result data structures |
| `YOLOTask.swift` | 52 | Task enum |
| `VideoCapture.swift` | 307 | AVFoundation camera pipeline |
| `BoundingBoxView.swift` | — | UIKit box rendering |
| `Plot.swift` | — | Pose skeleton + OBB drawing |
| `NonMaxSuppression.swift` | — | NMS algorithm |
| `ThresholdProvider.swift` | — | CoreML feature provider |
| `YOLOModelDownloader.swift` | — | Remote model download |
| `YOLOModelCache.swift` | — | Model caching |

### Problems

1. **`YOLOView` is a 1400-line god-view** — camera, inference, rendering, sliders, toolbar, zoom, photo capture, all tasks. SDK users can't use inference without this UI.
2. **`YOLOCamera` is a fake SwiftUI view** — 83 lines wrapping UIKit via `UIViewRepresentable`. Zero native SwiftUI.
3. **No headless real-time API** — can't stream camera results without `YOLOView`.
4. **Callback concurrency** — `DispatchQueue` + completion handlers everywhere. 5+ `@unchecked Sendable` hacks.
5. **Framework bakes in rendering** — `BoundingBoxView`, `Plot.swift`, mask overlays all in the SDK.
6. **5 app targets for 2 concepts** — 4 example apps + 1 demo app. UIKit examples add no value in 2026.

---

## Technical Decisions (from 2026 best practices research)

| Decision | Rationale |
|----------|-----------|
| **swift-tools-version: 6.0** | Strict concurrency enforced by default. No `swiftSettings` needed. |
| **Swift 6.2 `@concurrent`** for inference | New attribute for background work. Model loading and prediction must not block MainActor. |
| **`@MainActor` default isolation** | Swift 6.2 "Approachable Concurrency". All UI code runs on main actor by default. |
| **CoreML async prediction API** | `await model.prediction(input:)` — thread-safe, cancellable, supports concurrent ANE predictions. Replaces `VNImageRequestHandler.perform()` + `DispatchQueue`. |
| **`AsyncStream<YOLOResult>`** for real-time | Established 2025/2026 pattern for AVFoundation → Swift concurrency bridge. |
| **`UIViewRepresentable` only for camera preview** | No native SwiftUI camera API exists. This is the one unavoidable UIKit bridge. |
| **IOSurface-backed pixel buffers** | Avoids memory copies on Apple Silicon unified memory (ANE ↔ CPU ↔ GPU). |
| **YOLO26 NMS-free as default** | End-to-end model, no post-processing NMS. Confidence threshold only. 43% faster CPU inference. |
| **Keep NMS for legacy models** | Auto-detect via model metadata. YOLO11 models still work. |
| **W8A8 INT8 quantization docs** | Significant ANE latency gains on A17 Pro / M4. Document as recommended export option. |

---

## Target Architecture

```
Sources/
├── YOLOCore/                        ← Headless inference engine (no UIKit views)
│   ├── YOLO.swift                   # async model loading + single-image inference (UIImage + CGImage + CIImage)
│   ├── YOLOSession.swift            # real-time camera → AsyncStream<YOLOResult>
│   ├── YOLOConfiguration.swift      # thresholds, model options
│   ├── YOLOTask.swift               # task enum (moved from YOLO/)
│   ├── YOLOResult.swift             # result types (CGImage, not UIImage)
│   ├── Predictors/
│   │   ├── Predictor.swift          # protocol (stripped of callback listeners)
│   │   ├── BasePredictor.swift      # CoreML async prediction
│   │   ├── ObjectDetector.swift     # detection post-processing
│   │   ├── Classifier.swift         # classification post-processing
│   │   ├── Segmenter.swift          # segmentation post-processing
│   │   ├── PoseEstimator.swift      # pose post-processing
│   │   └── ObbDetector.swift        # OBB post-processing
│   ├── Camera/
│   │   └── CameraProvider.swift     # AVFoundation capture → AsyncStream<CVPixelBuffer>
│   ├── ModelLoading/
│   │   ├── ModelDownloader.swift    # remote download (async)
│   │   └── ModelCache.swift         # caching
│   └── PostProcessing/
│       ├── NonMaxSuppression.swift  # legacy NMS (YOLO11 compat)
│       └── ThresholdProvider.swift  # confidence thresholding
│
└── YOLOUI/                          ← SwiftUI views (depends on YOLOCore)
    ├── YOLOCamera.swift             # SwiftUI camera view (CameraPreview + YOLOSession + overlay)
    ├── CameraPreview.swift          # UIViewRepresentable for AVCaptureVideoPreviewLayer (~30 lines)
    ├── Overlays/
    │   ├── DetectionOverlay.swift   # SwiftUI Canvas bounding boxes
    │   ├── SegmentationOverlay.swift# SwiftUI Canvas masks
    │   ├── PoseOverlay.swift        # SwiftUI Canvas skeletons
    │   ├── OBBOverlay.swift         # SwiftUI Canvas rotated boxes
    │   └── ClassificationBanner.swift
    └── Controls/
        └── ThresholdControls.swift  # optional sliders bound to YOLOConfiguration

ExampleApps/
└── YOLOExample/                     ← 1 SwiftUI app (replaces 4 current example apps)

YOLOiOSApp/                          ← App Store app (SwiftUI native, adopts new SDK)
    └── Models/{Detect,Classify,Segment,Pose,OBB}Models/  ← drop-in YOLO26 .mlmodelc files
```

No `Sources/YOLO/` umbrella module. No UIKit views. No `YOLOView.swift`. Two targets, clean separation.

### Package.swift

```swift
// swift-tools-version: 6.0

let package = Package(
  name: "YOLO",
  platforms: [.iOS(.v16)],
  products: [
    .library(name: "YOLOCore", targets: ["YOLOCore"]),
    .library(name: "YOLOUI", targets: ["YOLOUI"]),
  ],
  dependencies: [
    .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.19"),
  ],
  targets: [
    .target(name: "YOLOCore", dependencies: ["ZIPFoundation"]),
    .target(name: "YOLOUI", dependencies: ["YOLOCore"]),
    .testTarget(name: "YOLOTests", dependencies: ["YOLOCore", "YOLOUI"], resources: [.process("Resources")]),
  ]
)
```

Consumers choose: `import YOLOCore` (headless engine) or `import YOLOUI` (SwiftUI views + engine).

---

## API Design

### Model loading

```swift
let model = try await YOLO("yolo26n", task: .detect)
let model = try await YOLO(url: remoteURL, task: .detect)
```

### Single-image inference

```swift
let result = model(uiImage)                 // UIImage (iOS-only, YOLOCore)
let result = model(cgImage)                 // CGImage (YOLOCore)
let result = model(ciImage)                 // CIImage (YOLOCore)
let result = try await model.predict(image) // async variant
```

### Real-time camera stream

```swift
let session = try await YOLOSession(model: model, camera: .back)

for await result in session.results {  // AsyncStream<YOLOResult>
    print(result.boxes)
}

session.pause()
session.resume()
session.stop()

session.configuration.confidenceThreshold = 0.3
session.configuration.iouThreshold = 0.5  // legacy models only
session.configuration.maxDetections = 30
```

### SwiftUI camera view

```swift
// Minimal — uses built-in overlays
YOLOCamera(model: "yolo26n", task: .detect)

// Custom result handling
YOLOCamera(model: "yolo26n", task: .detect) { result in
    // build your own UI from result
}

// Fully custom — no YOLOCamera, just the engine
let session = try await YOLOSession(model: model, camera: .back)
for await result in session.results { ... }
```

---

## Implementation Phases

### Phase 1: Extract `YOLOCore` (engine separation) — `engine-impl`

Separate inference from UI. This is the critical path — everything depends on it.

| Step | Action | Files |
|------|--------|-------|
| 1.1 | Create `Sources/YOLOCore/` directory structure | New dirs |
| 1.2 | Move predictors (`Predictor.swift`, `BasePredictor`, `ObjectDetector`, `Classifier`, `Segmenter`, `PoseEstimator`, `ObbDetector`, `NonMaxSuppression`, `ThresholdProvider`) to `YOLOCore/Predictors/` — replace `ResultsListener`/`InferenceTimeListener` callback protocols with `AsyncStream` (they become dead code) | Move + edit |
| 1.3 | Move `YOLOResult.swift`, `YOLOTask.swift` to `YOLOCore/` | Move |
| 1.4 | Replace `UIImage` with `CGImage` in `YOLOResult`. Keep `UIImage` `callAsFunction` overload in `YOLO.swift` (iOS-only package, acceptable) | Edit |
| 1.5 | Move `YOLOModelDownloader.swift`, `YOLOModelCache.swift` to `YOLOCore/ModelLoading/` | Move |
| 1.6 | Rewrite `VideoCapture.swift` → `YOLOCore/Camera/CameraProvider.swift`: strip UI delegates, output `AsyncStream<CVPixelBuffer>`, use IOSurface-backed buffers | Rewrite |
| 1.7 | Rewrite `YOLO.swift` init: `async/await` instead of completion callbacks | Rewrite |
| 1.8 | Rewrite `BasePredictor.create()`: `async` instead of `DispatchQueue` + callback. Use CoreML async prediction API. | Rewrite |
| 1.9 | Create `YOLOConfiguration.swift` — extract threshold logic from `YOLO` + `BasePredictor` into one place | New |
| 1.10 | Remove all `@unchecked Sendable` — make result types properly `Sendable` (they're value types) | Edit |
| 1.11 | Update `Package.swift` with `YOLOCore` target | Edit |
| 1.12 | **Delete** `Sources/YOLO/` entirely — all old copies of moved files, plus `YOLOView.swift`, `YOLOCamera.swift`, `BoundingBoxView.swift`, `Plot.swift` | Delete |

**Principle check:** Steps 1.2-1.6 are moves, not copies. Old files MUST be deleted (step 1.12). The entire `Sources/YOLO/` directory is replaced by `Sources/YOLOCore/` — no umbrella module, no UIKit views. Step 1.9 extracts threshold logic — the duplicate logic in `YOLO.swift` and `BasePredictor` must be removed.

### Phase 2: YOLO26 NMS-free support — `engine-impl`

Do this alongside Phase 1 since it simplifies post-processing.

| Step | Action |
|------|--------|
| 2.1 | Add model metadata detection in `BasePredictor`: read `creatorDefinedKey` to determine NMS-free (YOLO26) vs. NMS-required (YOLO11) |
| 2.2 | Update `ObjectDetector`: confidence threshold only for NMS-free models, skip NMS entirely |
| 2.3 | Update `Segmenter`, `PoseEstimator`, `ObbDetector` similarly |
| 2.4 | Keep `NonMaxSuppression.swift` for YOLO11 backwards compat |
| 2.5 | `YOLOConfiguration.iouThreshold` — only applies when model requires NMS |

**Principle check:** Don't add new NMS detection code if existing metadata parsing already handles it. Check `BasePredictor`'s existing `creatorDefinedKey` parsing first.

### Phase 3: Create `YOLOSession` (real-time streaming) — `engine-impl`

| Step | Action |
|------|--------|
| 3.1 | Create `YOLOSession.swift` in `YOLOCore/` — wraps `CameraProvider` + predictor |
| 3.2 | Expose `results` as `AsyncStream<YOLOResult>` |
| 3.3 | `pause()`, `resume()`, `stop()` lifecycle |
| 3.4 | `configuration` property (mutable thresholds) |
| 3.5 | `latestResult` for SwiftUI `@Observable` binding |
| 3.6 | Camera permissions: throw descriptive errors |
| 3.7 | Mark inference with `@concurrent` (Swift 6.2) to run off MainActor |

### Phase 4: Create `YOLOUI` (SwiftUI views) — `ui-impl`

Blocked by Phase 1. Only start after `YOLOCore` compiles.

| Step | Action |
|------|--------|
| 4.1 | Create `Sources/YOLOUI/` |
| 4.2 | `CameraPreview.swift` — `UIViewRepresentable` for `AVCaptureVideoPreviewLayer` only (~30 lines, no inference) |
| 4.3 | `DetectionOverlay.swift` — SwiftUI `Canvas` bounding boxes from `[Box]` |
| 4.4 | `SegmentationOverlay.swift` — SwiftUI `Canvas` masks |
| 4.5 | `PoseOverlay.swift` — SwiftUI `Canvas` skeletons |
| 4.6 | `OBBOverlay.swift` — SwiftUI `Canvas` rotated boxes |
| 4.7 | `ClassificationBanner.swift` — SwiftUI label |
| 4.8 | `YOLOCamera.swift` — composes CameraPreview + YOLOSession + auto-overlay by task |
| 4.9 | `ThresholdControls.swift` — optional sliders bound to `YOLOConfiguration` |
| 4.10 | Update `Package.swift` with `YOLOUI` target |

**Principle check:** Each overlay must be minimal. Don't build a generic "overlay framework" — 5 simple views, one per task. If two overlays share box-drawing logic, extract the minimum shared code, don't create an abstraction layer.

### Phase 5: Consolidate example apps — `ui-impl`

| Step | Action |
|------|--------|
| 5.1 | Create `ExampleApps/YOLOExample/` SwiftUI app |
| 5.2 | `CameraView.swift` — real-time with task picker (all 5 tasks) |
| 5.3 | `PhotoView.swift` — single-image from photo library |
| 5.4 | Bundle `yolo26n.mlmodelc` (detection) as default model |
| 5.5 | **Delete** `ExampleApps/YOLORealTimeSwiftUI/`, `YOLORealTimeUIKit/`, `YOLOSingleImageSwiftUI/`, `YOLOSingleImageUIKit/` entirely |

**Principle check:** Step 5.5 is non-negotiable. 4 old apps deleted, 1 new app created. Net -3 app targets.

### Phase 6: Model drop-in + App Store app (SwiftUI rewrite) — `ui-impl`

| Step | Action |
|------|--------|
| 6.1 | **Rewrite `YOLOiOSApp` in SwiftUI** — replace all UIKit views with SwiftUI equivalents using YOLOUI components |
| 6.2 | Standardize `YOLOiOSApp/Models/{Detect,Classify,Segment,Pose,OBB}Models/` directory structure |
| 6.3 | Auto-discovery: scan model directories at launch, populate picker dynamically (delete hardcoded model lists) |
| 6.4 | Support `.mlmodelc` (pre-compiled) and `.mlpackage` (runtime-compiled) |
| 6.5 | Bundle YOLO26 nano models for all 5 tasks as defaults |
| 6.6 | Model validation on load — clear errors for task/model mismatches |
| 6.7 | Model hot-swap at runtime without restarting camera session |
| 6.8 | Ensure App Store compliance: signing, entitlements, privacy manifests |
| 6.9 | Document drop-in workflow: `ultralytics` Python export → drag into Xcode → build and run |

**Principle check:** Step 6.1 is key — no UIKit views in the final app. Old UIKit files (`YOLOView.swift`, `BoundingBoxView.swift`, `Plot.swift`) are already deleted in Phase 1 step 1.12.

### Phase 7: Swift 6 concurrency audit — `reviewer` + all agents

This runs continuously, not as a separate phase.

| Check | Action |
|-------|--------|
| Zero `@unchecked Sendable` | All result types are value types → naturally Sendable |
| Zero `DispatchQueue.main.async` | Use `@MainActor` |
| Zero completion handler callbacks | Use `async/await` |
| `@concurrent` on inference paths | Model loading, prediction, post-processing |
| CoreML async prediction API | `await model.prediction()` everywhere |
| No UIKit views anywhere | SwiftUI only. UIViewRepresentable only for AVCaptureVideoPreviewLayer |
| No UIKit imports in `YOLOCore` | Except `AVFoundation` for camera |

---

## What stays

- **5 tasks** — detect, classify, segment, pose, obb
- **CoreML backend** — right choice for iOS, no alternatives needed
- **Task-specific predictors** — inheritance hierarchy is sound
- **Model download + cache** — just needs async/await wrappers
- **`callAsFunction` pattern** — `model(image)` is elegant
- **`YOLOiOSApp/`** — App Store app, adopts new SDK
- **ZIPFoundation** — sole external dependency
- **5 model sizes per task** — nano through extra-large

## What gets deleted

- **`YOLOView.swift`** (1413 lines) — god-view, deleted entirely (not deprecated)
- **`YOLOCamera.swift`** (83 lines) — fake SwiftUI wrapper, replaced by real SwiftUI `YOLOCamera`
- **`BoundingBoxView.swift`** — replaced by `DetectionOverlay.swift`
- **`Plot.swift`** — replaced by `PoseOverlay.swift` + `OBBOverlay.swift`
- **`Sources/YOLO/`** umbrella module — no longer exists, consumers use `YOLOCore` or `YOLOUI` directly
- **4 example apps** — replaced by 1 `YOLOExample`
- All `@unchecked Sendable` — proper concurrency
- All `DispatchQueue` threading — async/await + actors
- All YOLO11 references in code and docs — YOLO26 default
- All UIKit views — SwiftUI only

## Default bundled models

| Task | Model | File | NMS |
|------|-------|------|-----|
| Detect | YOLO26 nano | `yolo26n.mlmodelc` | No (NMS-free) |
| Classify | YOLO26 nano | `yolo26n-cls.mlmodelc` | N/A |
| Segment | YOLO26 nano | `yolo26n-seg.mlmodelc` | No (NMS-free) |
| Pose | YOLO26 nano | `yolo26n-pose.mlmodelc` | No (NMS-free) |
| OBB | YOLO26 nano | `yolo26n-obb.mlmodelc` | No (NMS-free) |

Users drop in `s`, `m`, `l`, `x` variants. Auto-discovered by model picker.

## Migration

| Current | New | Breaking? |
|---------|-----|-----------|
| `import YOLO` | `import YOLOCore` or `import YOLOUI` | Yes (rename) |
| `YOLO("m", task:) { cb }` | `try await YOLO("m", task:)` | Yes |
| `model(image)` | Same | No |
| `YOLOCamera(modelPathOrName:)` | `YOLOCamera(model:)` | Rename |
| `YOLOView` | **Deleted** — use `YOLOCamera` (SwiftUI) or `YOLOSession` (headless) | Yes |
| `YOLOResult.annotatedImage: UIImage?` | `.annotatedImage: CGImage?` | Yes |
| YOLO11 models | Auto-detected, NMS applied | No |
| YOLO26 models | Default, confidence threshold only | No |

This is a major version bump. Clean breaks are better than deprecated shims that accumulate forever.

## Phase dependencies and assignment

```
Phase 1 (YOLOCore)          → engine-impl  ← CRITICAL PATH, start first
Phase 2 (YOLO26 NMS-free)   → engine-impl  ← do alongside Phase 1
Phase 3 (YOLOSession)        → engine-impl  ← after Phase 1
Phase 4 (YOLOUI)             → ui-impl      ← blocked by Phase 1
Phase 5 (Example apps)       → ui-impl      ← blocked by Phase 4
Phase 6 (Drop-in + App Store)→ ui-impl      ← blocked by Phase 4
Phase 7 (Concurrency audit)  → reviewer     ← continuous
```

`engine-impl` starts immediately on Phases 1+2+3. `ui-impl` starts as soon as Phase 1 is complete. `reviewer` reviews continuously. Only 2 implementers + 1 reviewer — simpler team, less coordination overhead.
