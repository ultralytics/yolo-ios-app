# Claude Prompt — YOLO iOS SDK Modernization

Copy everything below the line and paste it as your prompt to Claude Code.

---

Read `PLAN.md` at repo root. It contains the complete modernization plan for this YOLO iOS SDK, including architecture, phases, API design, and agent team instructions.

## What you're building

Rewrite this iOS SDK from a UIKit-based YOLO11 demo into a modern Swift 6 / SwiftUI / YOLO26 framework. Two Package targets (`YOLOCore` headless engine + `YOLOUI` SwiftUI views), no UIKit views, no umbrella module. The App Store app (`YOLOiOSApp`) must be rewritten in SwiftUI and look identical to the current app (same sliders, toolbar, overlays, camera controls). All 5 tasks (detect, classify, segment, pose, OBB) must work with YOLO26 NMS-free models.

## How to execute

1. Read `PLAN.md` fully — it has Core Principles, target architecture, Package.swift, API design, 7 implementation phases, deletion lists, and migration tables.

2. Spawn a team following the Agent Team Instructions in PLAN.md:
   - Create team `yolo-modernize` with TeamCreate
   - Create tasks for all 7 phases with TaskCreate
   - Spawn `engine-impl` (Phases 1, 2, 3 — YOLOCore engine)
   - Spawn `ui-impl` (Phases 4, 5, 6 — YOLOUI views + example app + App Store app), blocked by Phase 1
   - Spawn `reviewer` (Phase 7 — continuous concurrency/principles audit)

3. Every agent must follow the 7 Core Principles from PLAN.md. The reviewer must reject violations.

4. When all phases complete, verify:
   - `swift build` succeeds with swift-tools-version 6.0 (zero warnings)
   - Zero `@unchecked Sendable`, zero `DispatchQueue`, zero completion callbacks
   - No UIKit views anywhere (only `UIViewRepresentable` for `AVCaptureVideoPreviewLayer`)
   - `Sources/YOLO/` directory is deleted entirely
   - All 4 old example apps deleted, 1 new `YOLOExample` created
   - `YOLOiOSApp` runs in SwiftUI with same UI as current app
   - YOLO26 NMS-free models work for all 5 tasks
   - YOLO11 legacy models still work (auto-detected, NMS applied)

Start now. Read PLAN.md first, then spawn the team.
