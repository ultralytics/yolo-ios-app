# AGENTS.md

Repository guidance for coding agents. `CLAUDE.md` is a symlink to this file.

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

## Commands and validation

```bash
bash scripts/download-models.sh
xcrun simctl list devices available
xcodebuild -scheme UltralyticsYOLO -sdk iphonesimulator -derivedDataPath Build/ \
  -destination "platform=iOS Simulator,id=<SIMULATOR_UDID>,arch=arm64" \
  IPHONEOS_DEPLOYMENT_TARGET=16.0 build test
```

Use Xcode and an available simulator UDID; `swift test` cannot build this UIKit package. Add `-only-testing:YOLOTests/PlotTests` for focused tests. Model resources must exist even to build the test target. Camera changes need a physical device. CI also runs strict Periphery against the app project; see `.github/workflows/ci.yml`. Keep `swift-tools-version: 5.10` and iOS 13 compatibility in the SDK, guarding newer APIs even though CI and the app target iOS 16.

## Where to look

- Model loading and decoding → `Sources/UltralyticsYOLO/BasePredictor.swift`, `ModelPathResolver.swift`, and the task predictor.
- Camera and overlay geometry → `Sources/UltralyticsYOLO/VideoCapture.swift`, `YOLOView.swift`, and `Plot.swift`.
- App integration → `YOLOiOSApp/`.
- Tests and model resources → `Tests/YOLOTests/`.
- Model download and export → `scripts/download-models.sh`, `scripts/export-models.py`.
- Measured native configuration → `docs/performance.md`.
- Public API docs → `Sources/UltralyticsYOLO/README.md`.

## Conventions

- License header `// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` on every source file (`#` form for shell/Python/YAML) — Ultralytics Actions adds it automatically; don't add or revert it manually.
- Formatting is enforced by `format.yml` pushing commits onto PRs (swift-format, Prettier, shfmt, codespell, Ruff for Python) — pull its commits instead of re-formatting locally.
- Tests are XCTest in `Tests/YOLOTests`; model-backed tests load `.mlpackage` bundles from test resources (run the download script first) and none hit the live network.
- Releases: bump `MARKETING_VERSION` (two build configurations in `project.pbxproj`) and `s.version` in `UltralyticsYOLO.podspec` together in the release PR; merging to `main` then auto-tags, releases, and publishes the pod. The SPM `from:` version in `README.md`, `README.zh-CN.md`, and `Sources/UltralyticsYOLO/README.md` is refreshed to the latest released tag alongside. `ultralytics/yolo-flutter-app` consumes the pod (its `ios/ultralytics_yolo.podspec` pins `'>= 8.9.14', '< 9.0'` at the time of writing), so a public API break there is a `9.0` change.
- Archive app builds auto-bump `CFBundleVersion` in `YOLOiOSApp/YOLOiOSApp/Info.plist` (a Run Script phase gated on `ACTION == install`) — never commit a stray build-number bump.
- `README.md` and `README.zh-CN.md` are translations of each other — apply any README change to both. `Sources/UltralyticsYOLO/README.md` is the package README (API usage, asset tables) and `docs/performance.md` is the canonical profiling record; update them when public API or shipped configuration changes.
- Task order in user-facing tables, `appTasks`, `remoteModelsInfo`, the export matrix, and the download script is `detect, segment, semantic, depth, classify, pose, obb`.
- Model naming: `yolo26{n,s,m,l,x}{"",-seg,-sem,-depth,-cls,-pose,-obb}`; official assets are int8 `.mlpackage.zip` archives at `https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/<model>.mlpackage.zip`, 224×224 input for classify and 640×640 for every other task, exported with `nms=False`.

## Pitfalls

- `useGpu` is a misnomer kept for Flutter parity: `true` means ANE + CPU (`.cpuAndNeuralEngine` on iOS 16+, `.all` on iOS 13–15); the GPU is deliberately excluded on iOS 16+. Preserve the iOS 13–15 fallback; do not switch the iOS 16+ branch to `.all` (measured slower/jitterier, `docs/performance.md`).
- YOLO26 (NMS-free) models ignore `iouThreshold` — the provider forces `1.0` and the end2end decoders skip NMS — so IoU-slider bugs only reproduce with YOLO11 or `nms=None` exports. `confidenceThreshold` reaches the Vision NMS pipeline only through the `ThresholdProvider`; raw-tensor decoders filter it in Swift.
- Two independent on-device model caches: SDK `YOLOModelCache` (`Library/Caches/YOLOModels`, SHA-256 keys) and app `ModelCacheManager` (`Documents/<key>-mobile-standard-v1.mlmodelc`). Replacing release assets in place without bumping the revision strings leaves devices on stale models.
