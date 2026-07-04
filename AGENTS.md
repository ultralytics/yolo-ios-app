# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# One-time: download the six nano Core ML models (required by model-backed tests;
# also copies them into YOLOiOSApp/Models/ for the app bundle)
bash scripts/download-models.sh

# Run all package tests (mirrors .github/workflows/ci.yml; get a simulator UDID
# from `xcrun simctl list devices available` — use id=, name= resolves unreliably)
xcodebuild -scheme UltralyticsYOLO -sdk iphonesimulator -derivedDataPath Build/ \
  -destination "platform=iOS Simulator,id=<SIMULATOR_UDID>,arch=arm64" \
  IPHONEOS_DEPLOYMENT_TARGET=16.0 build test

# Run a single test class or method: append e.g.
#   -only-testing:YOLOTests/PlotTests
#   -only-testing:YOLOTests/PlotTests/testUltralyticsColorsExist

# Coverage as CI runs it: add `-enableCodeCoverage YES clean` to the command above;
# ci.yml then exports lcov with llvm-cov and filters out camera/UI files before Codecov upload

# Format (what format.yml auto-applies to PRs; no .swift-format config file = defaults)
swift-format --in-place --recursive .     # brew install swift-format
npx prettier --write "**/*.{md,yml,json}" # YAML/JSON/Markdown

# Dead-code check (CI `periphery` job, strict; brew install periphery)
periphery scan --project YOLOiOSApp/YOLOiOSApp.xcodeproj --schemes YOLOiOSApp \
  --exclude-tests --retain-public --report-include 'Sources/UltralyticsYOLO/**/*.swift' \
  --strict -- -destination "platform=iOS Simulator,id=<SIMULATOR_UDID>,arch=arm64"

# Model export env (scripts/export-models.py; needs a sibling ultralytics checkout)
uv venv --python 3.13 .venv && uv pip install -e "../ultralytics[export]"
```

CI (`ci.yml`) runs two jobs on `macos-26`: `test` (build + test + Codecov with `fail_ci_if_error: true`) and `periphery` (dead-code scan, `--strict` fails on any unused declaration). `Package.swift` is pinned to `swift-tools-version: 5.10` for CI compatibility — do not raise it.

## Architecture

- Single SPM library target `UltralyticsYOLO` (`Sources/UltralyticsYOLO/`), also published as the `UltralyticsYOLO` CocoaPod; the `ultralytics/yolo-flutter-app` plugin depends on the pod (pinned `< 9.0`), so public API breaks there too. Package floor is iOS 13 (with `@available` fallbacks) while the main app `YOLOiOSApp/` targets iOS 16.
- Zero third-party dependencies: ZIP extraction of downloaded models is the in-repo `MiniZip.swift` (Foundation + Compression only).
- Inference flow: `YOLO.swift` facade (`callAsFunction` overloads for URL/String/UIImage/CIImage/CGImage) → `BasePredictor` subclasses (`ObjectDetector`, `Segmenter`, `SemanticSegmenter`, `Classifier`, `PoseEstimator`, `ObbDetector`) → Vision `VNCoreMLRequest`. `YOLOView` (UIKit, wraps `AVCaptureSession` + overlays) and `YOLOCamera` (SwiftUI) provide real-time camera UI.
- YOLO26 vs YOLO11: model metadata key `nms == "false"` marks NMS-free YOLO26 end2end models (detect output `[1, 300, 6]` xyxy pixel coords, decoded in Swift); default `requiresNMS = true` keeps the Core ML NMS path for YOLO11 (`[1, 4+nc, 8400]` xywh). Always index `MLMultiArray` via `strides`.
- `.mlpackage` models are never committed (gitignored); tests and the app get them from the `v8.3.0` release assets via `scripts/download-models.sh` (an Xcode "Download YOLO Models" build phase runs it locally and is skipped on GitHub Actions, where CI runs the script as its own step).
- Publishing (`publish.yml`, push to `main`, runs only when the pushing actor is `glenn-jocher`): a new `MARKETING_VERSION` in `YOLOiOSApp/YOLOiOSApp.xcodeproj/project.pbxproj` triggers tag `v{version}` + GitHub release + `pod trunk push` + a squashed `testflight` branch force-pushed for Xcode Cloud; an unchanged version still ships a TestFlight build.

## Conventions

- License header `// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` on every source file — Ultralytics Actions adds it automatically; don't add or revert it manually.
- Formatting is enforced by `format.yml` pushing commits onto PRs (swift-format, Prettier, codespell, Ruff/docformatter for Python) — pull its commits instead of re-formatting locally.
- Tests are XCTest in `Tests/YOLOTests`; model-backed tests load `.mlpackage` bundles from test resources (run the download script first) and none hit the live network.
- Releases: bump `MARKETING_VERSION` (two build configurations in `project.pbxproj`) and `s.version` in `UltralyticsYOLO.podspec` together in the release PR; merging to `main` then auto-tags, releases, and publishes the pod.
- Archive app builds auto-bump `CFBundleVersion` in `YOLOiOSApp/Info.plist` — never commit a stray build-number bump.
- `README.md` and `README.zh-CN.md` are translations of each other — apply any README change to both.
