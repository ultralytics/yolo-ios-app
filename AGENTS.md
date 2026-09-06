# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# One-time: download the seven nano Core ML models (required by model-backed tests;
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

# Model export env (scripts/export-models.py)
uv venv --python 3.13 .venv && uv pip install "ultralytics[export-coreml]>=8.4.142"
```

CI (`ci.yml`) runs two jobs on `macos-26`: `test` (build + test + a non-blocking Codecov upload) and `periphery` (dead-code scan, `--strict` fails on any unused declaration). `Package.swift` is pinned to `swift-tools-version: 5.10` for CI compatibility — do not raise it.

## Architecture

- Single SPM library target `UltralyticsYOLO` (`Sources/UltralyticsYOLO/`), also published as the `UltralyticsYOLO` CocoaPod; the `ultralytics/yolo-flutter-app` plugin depends on the pod (pinned `< 9.0`), so public API breaks there too. Package floor is iOS 13 (with `@available` fallbacks) while the main app `YOLOiOSApp/` targets iOS 16.
- Zero third-party dependencies: ZIP extraction of downloaded models is the in-repo `MiniZip.swift` (Foundation + Compression only).
- Inference flow: `YOLO.swift` facade (`callAsFunction` overloads for URL/String/UIImage/CIImage/CGImage) → `BasePredictor` subclasses (`ObjectDetector`, `Segmenter`, `SemanticSegmenter`, `DepthEstimator`, `Classifier`, `PoseEstimator`, `ObbDetector`) → Vision `VNCoreMLRequest`. `YOLOView` (UIKit, wraps `AVCaptureSession` + overlays) and `YOLOCamera` (SwiftUI) provide real-time camera UI.
- Export scripts require `ultralytics>=8.4.142`: Core ML uses `nms=False` for NMS-free YOLO26 outputs; `nms=None` exports raw one-to-many outputs and `nms=True` embeds NMS where supported. `end2end` describes graph metadata; use `nms` to configure exports. Predictors decode Vision NMS observations or raw tensors by their actual layouts. Always index `MLMultiArray` via `strides`.
- `.mlpackage` models are never committed (gitignored); tests and the app get them from the `v8.3.0` release assets via `scripts/download-models.sh` (an Xcode "Download YOLO Models" build phase runs it locally and is skipped on GitHub Actions, where CI runs the script as its own step).
- Publishing (`publish.yml`, push to `main`, runs only when the pushing actor is `glenn-jocher`): a new `MARKETING_VERSION` in `YOLOiOSApp/YOLOiOSApp.xcodeproj/project.pbxproj` triggers tag `v{version}` + GitHub release + `pod trunk push` + a squashed `testflight` branch force-pushed for Xcode Cloud; an unchanged version still ships a TestFlight build.

## Conventions

- License header `// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` on every source file — Ultralytics Actions adds it automatically; don't add or revert it manually.
- Formatting is enforced by `format.yml` pushing commits onto PRs (swift-format, Prettier, codespell, Ruff/docformatter for Python) — pull its commits instead of re-formatting locally.
- Tests are XCTest in `Tests/YOLOTests`; model-backed tests load `.mlpackage` bundles from test resources (run the download script first) and none hit the live network.
- Releases: bump `MARKETING_VERSION` (two build configurations in `project.pbxproj`) and `s.version` in `UltralyticsYOLO.podspec` together in the release PR; merging to `main` then auto-tags, releases, and publishes the pod.
- Archive app builds auto-bump `CFBundleVersion` in `YOLOiOSApp/YOLOiOSApp/Info.plist` — never commit a stray build-number bump.
- `README.md` and `README.zh-CN.md` are translations of each other — apply any README change to both.
