<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 适用于 iOS 的 Ultralytics YOLO：App 与 Swift Package

[![Ultralytics Actions](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml)
[![CI](https://github.com/ultralytics/yolo-ios-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-ios-app/branch/main/graph/badge.svg)](https://codecov.io/gh/ultralytics/yolo-ios-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

欢迎来到 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) iOS App 的 GitHub 仓库！📖 这个项目可将你的 iOS 设备变成强大的[实时推理](https://www.ultralytics.com/glossary/real-time-inference)工具，支持[目标检测](https://www.ultralytics.com/glossary/object-detection)、实例分割、语义分割、图像分类、姿态估计以及旋转框检测。该 SDK 同时支持传统的 [YOLO11](https://docs.ultralytics.com/models/yolo11)（使用 Core ML NMS）和最新的 [YOLO26 模型](https://platform.ultralytics.com/ultralytics/yolo26)（无 NMS，使用 Swift 侧后处理）。你可以直接从 [App Store](https://apps.apple.com/cn/app/ultralytics-yolo/id1452689527) 下载应用，也可以参考本指南，将 YOLO 能力集成到你自己的 Swift 应用中。

<div align="center">
  <br>
  <a href="https://apps.apple.com/cn/app/ultralytics-yolo/id1452689527" target="_blank"><img width="100%" src="https://github.com/user-attachments/assets/d5dab2e7-f473-47ce-bc63-69bef89ba52a" alt="Ultralytics YOLO iOS App previews"></a>
  <br>
  <br>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
  <br>
  <br>
  <a href="https://apps.apple.com/cn/app/ultralytics-yolo/id1452689527" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/app-store.svg" width="15%" alt="Apple App store"></a>
</div>

## ✨ 为什么选择原生 YOLO iOS？

- 原生 iOS 性能 - 通过 Swift 与 Core ML 获得最高速度
- 针对 Apple Silicon 优化 - 充分利用 Neural Engine 与 GPU
- 实时推理 - 在最新款 iPhone 上达到相机帧率（约 30 FPS）的性能
- 低延迟 - 无框架额外开销，直接访问硬件能力
- iOS 优先设计 - 原生 UI/UX，遵循 Apple 设计规范
- Core ML 集成 - 使用 Apple 官方优化的机器学习框架
- 同时支持 YOLO26（无 NMS）与 YOLO11 模型
- 零依赖 - 纯 Swift，仅依赖 Apple 官方框架；无任何第三方 package

| 功能                  | iOS | 详细说明                     |
| --------------------- | --- | ---------------------------- |
| 目标检测              | ✅  | 边界框、类别标签和置信度分数 |
| 实例分割              | ✅  | 实例掩膜、边界框和类别       |
| 语义分割              | ✅  | 密集逐像素类别图             |
| 图像分类              | ✅  | 最高类别预测和分数           |
| 姿态估计              | ✅  | 关键点、边界框和置信度分数   |
| 定向边界框（OBB）检测 | ✅  | 旋转框和多边形角点           |

## 📂 仓库内容

此仓库为在 Apple 平台上运行 YOLO 模型提供了完整方案，帮助你构建强大的[边缘 AI](https://www.ultralytics.com/glossary/edge-ai)能力：

### [**Ultralytics YOLO iOS App（主应用）**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp)

这是主要的 iOS 应用，可通过设备相机或图片库轻松进行实时 YOLO 推理。发布的应用打包了全部六个官方 nano Core ML 模型，更大的变体可按需下载；你也可以将自己的 [Core ML](https://developer.apple.com/documentation/coreml) 模型添加到应用工程中进行测试。

### [**Swift Package（YOLO 库）**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/UltralyticsYOLO)

这是一个面向 iOS 和 iPadOS 的轻量级 [Swift](https://developer.apple.com/swift/) package，用于简化 YOLO26 等 YOLO 模型在应用中的集成与使用。借助 [SwiftUI](https://developer.apple.com/xcode/swiftui/)，你可以用极少的代码轻松集成 YOLO：

```swift
// 对 UIImage 执行推理
let result = model(uiImage)
```

```swift
// 使用内置相机视图，对应用内打包的模型进行实时推理
var body: some View {
    YOLOCamera(
        modelPathOrName: "yolo26n-seg",
        task: .segment,
        cameraPosition: .back
    )
    .ignoresSafeArea()
}
```

## 📦 官方模型资源

官方模型以 GitHub 发布（release）资源的形式提供，而不是提交到仓库中的大文件。主 iOS 应用会在构建时下载六个 nano Core ML 资源并打包进应用；更大的应用模型、Swift package 的 `YOLO(url:)` 加载方式以及 Flutter package 资源会在首次使用时下载官方模型并缓存到本地。

主应用 YOLOiOSApp 会将**全部六个 nano 模型**（每个任务一个：检测、分割、语义分割、分类、姿态、OBB）打包进发布的应用中（包括 App Store/归档构建）。这些模型在构建时由运行 [`scripts/download-models.sh`](scripts/download-models.sh) 的 **Download YOLO Models** Xcode 构建阶段从 GitHub 发布资源下载——`.mlpackage` 文件**绝不会提交到仓库**（`*.mlpackage` 已被 gitignore 忽略）。该步骤是幂等的，在 GitHub Actions CI 上会被跳过，CI 会在单独的步骤中运行同一脚本。

| 运行时资源                    | 使用方                                          | 发布版本                                                                                         |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Core ML int8 `.mlpackage.zip` | iOS 应用、Swift package、iOS/macOS 上的 Flutter | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)         |
| TFLite int8 `.tflite`         | Android 上的 Flutter                            | [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5) |

URL 模式：

- Core ML：`https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/<model>.mlpackage.zip`
- TFLite：`https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5/<model>_int8.tflite`

iOS 应用的模型注册表是 [`RemoteModels.swift`](YOLOiOSApp/YOLOiOSApp/RemoteModels.swift)。它枚举了检测、分割、语义分割、分类、姿态和 OBB 任务的 YOLO26 `n/s/m/l/x` 资源，并将每个模型 ID 指向 `v8.3.0` Core ML 发布版本。下表中的 Core ML 列由本仓库维护；TFLite 列概述了 Flutter 仓库的 Android 导出脚本及其发布资源。

| 属性      | Core ML                            | TFLite                           |
| --------- | ---------------------------------- | -------------------------------- |
| 模型 ID   | `yolo26{n,s,m,l,x}`                | `yolo26{n,s,m,l,x}`              |
| 任务      | detect、seg、sem、cls、pose、obb   | detect、seg、sem、cls、pose、obb |
| 格式      | `.mlpackage.zip`                   | `.tflite`                        |
| `int8`    | `True`                             | `True`                           |
| `imgsz`   | 分类 `224`；OBB `1024`；其余 `640` | 分类 `224`；其余 `640`           |
| `nms`     | `False`                            | `False`                          |
| `end2end` | `True`                             | `False`                          |
| 校准      | 导出器默认值                       | 按任务的 `TASK2CALIBRATIONDATA`  |
| 后处理    | Swift/Core ML                      | Android 原生                     |

Core ML 资源使用 `nms=False` 和 `end2end=True` 导出：`nms=False` 会去掉 Core ML NMS 流水线，`end2end=True` 则提供由 Swift 解码器消费的 YOLO26 解码输出契约。TFLite 导出脚本同时传入 `nms=False` 和 `end2end=False`；`end2end=False` 会为 Android LiteRT 转换路径禁用 YOLO26 端到端头。

### Core ML 发布工作流

权威导出脚本是 [`scripts/export-models.py`](scripts/export-models.py)。它定义了任务/尺寸矩阵、导出图像尺寸、int8 Core ML 设置、`.mlpackage.zip` 打包、可选的本地应用复制步骤以及可选的 GitHub 发布上传。

```bash
uv venv --python 3.13 .venv
uv pip install -e "../ultralytics[export]"
uv run python scripts/export-models.py
```

常用变体：

```bash
# 仅导出 nano 任务模型用于本地验证，并将其复制到 YOLOiOSApp/Models/。
uv run python scripts/export-models.py --sizes n --copy-to-app

# 导出全部官方 Core ML 资源并上传到规范发布版本。
uv run python scripts/export-models.py --upload --repo ultralytics/yolo-ios-app --tag v8.3.0
```

该脚本从名为 `yolo26<size><suffix>.pt` 的检查点导出，例如 `yolo26n.pt`、`yolo26s-seg.pt`、`yolo26m-sem.pt`、`yolo26l-pose.pt` 和 `yolo26x-obb.pt`。在本 SDK 中 YOLO26 是无 NMS 的，因此官方 Core ML 资源使用 `nms=False` 和 `end2end=True` 导出；Swift 侧后处理负责处理检测、分割、姿态和 OBB 输出。

### Android TFLite 对应资源

Flutter package 使用的 Android 资源在 Flutter 仓库中维护，而不在本 iOS 仓库中。其规范导出脚本是 `ultralytics/yolo-flutter-app` 仓库中的 `scripts/export-tflite-models.py`；它将匹配的 YOLO26 任务/尺寸矩阵导出为 int8 `.tflite` 资源，使用按任务的 `ultralytics.cfg.TASK2CALIBRATIONDATA` 默认值进行校准，并上传到 `yolo-flutter-app` `v0.3.5`。

## 🛠️ 快速开始

如果你刚接触移动端 YOLO，或想快速测试自己的模型，建议先从主应用 YOLOiOSApp 开始。六个 nano 任务模型会在构建时打包进应用，因此应用安装后即可离线运行；更大的模型尺寸可按需下载。

- [**Ultralytics YOLO iOS App（主应用）**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp)：在 iOS 上体验 YOLO 推理的最简单方式。

如果你已经准备好将 YOLO 集成到自己的项目中，可以继续查看 Swift Package 和示例应用。

- [**Swift Package（YOLO 库）**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/UltralyticsYOLO)：将 YOLO 能力集成到你的 Swift 应用中。
- [**示例应用**](https://github.com/ultralytics/yolo-ios-app/tree/main/ExampleApps)：查看基于 YOLO Swift Package 的实际实现示例。

使用 Swift Package Manager 将 `UltralyticsYOLO` package 添加到你的应用：

```swift
.package(url: "https://github.com/ultralytics/yolo-ios-app.git", from: "8.9.0")
```

或使用 CocoaPods：

```ruby
pod 'UltralyticsYOLO', '~> 8.9'
```

然后 `import UltralyticsYOLO` 并使用 `YOLO` 类——完整用法请参阅 [Swift Package README](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/UltralyticsYOLO)。同一个 `UltralyticsYOLO` package 同时驱动本原生 iOS 应用和 [Ultralytics YOLO Flutter 插件](https://github.com/ultralytics/yolo-flutter-app)，在多个平台间保持单一事实来源。

## ✨ 核心亮点

- **实时推理**：使用优化后的 [Core ML 模型](https://docs.ultralytics.com/integrations/coreml)，在 iPhone 和 iPad 上实现高速、高精度的目标检测，并可结合[模型量化](https://www.ultralytics.com/glossary/model-quantization)等技术进一步提升性能。有关设备端性能分析以及相机/Core ML 配置的依据，请参阅 [docs/performance.md](docs/performance.md)。
- **Apple 移动平台支持**：Swift Package 面向 iOS 和 iPadOS，并提供原生 Core ML 集成。
- **灵活任务支持**：支持[目标检测](https://docs.ultralytics.com/tasks/detect)、[实例分割](https://docs.ultralytics.com/tasks/segment)、[语义分割](https://docs.ultralytics.com/tasks/semantic)、[分类](https://docs.ultralytics.com/tasks/classify)、[姿态估计](https://docs.ultralytics.com/tasks/pose)以及[旋转框（OBB）检测](https://docs.ultralytics.com/tasks/obb)。

## 🧪 测试流程

此仓库为 YOLO Swift Package 和示例应用都提供了较完整的[单元测试](https://en.wikipedia.org/wiki/Unit_testing)，以确保代码的可靠性与稳定性。

### 运行测试

测试依赖 Core ML 模型文件（`.mlpackage`），但由于文件体积较大，仓库中不会提交这些模型。若要执行带模型校验的测试，请先在仓库根目录运行与 CI 和应用构建阶段相同的下载脚本：

```bash
bash scripts/download-models.sh
```

该脚本会将六个 nano Core ML package 下载到 `Tests/YOLOTests/Resources/`，并复制到 `YOLOiOSApp/Models/<Task>/`，供主应用在构建时打包进应用。你也可以使用 [Ultralytics Python 库的导出功能](https://docs.ultralytics.com/modes/export) 导出或替换为自定义 Core ML 模型。如果某个测试 target 支持 `SKIP_MODEL_TESTS`，保持为 `true` 会跳过需要加载和运行模型的测试。

### 测试覆盖范围

- **YOLO Swift Package**：包含针对模型加载、预处理、推理以及不同任务后处理流程的核心功能测试。
- **示例应用**：包含对 UI 组件、模型集成以及示例应用中实时推理性能的测试。

### 测试文档

每个测试目录（例如 `Tests/YOLOTests`）都可能包含一个 `README.md`，用于说明该组件的具体测试方法，内容通常包括：

- 所需模型文件及其获取方式。
- 模型转换与配置步骤。
- 测试策略概览。
- 关键测试用例说明。

## 💡 参与贡献

我们非常欢迎你为开源项目贡献力量！你的支持将帮助我们持续推动[人工智能（AI）](https://www.ultralytics.com/glossary/artificial-intelligence-ai)的发展边界。欢迎查阅[贡献指南](https://docs.ultralytics.com/help/contributing)，也可以通过我们的[问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享你的反馈。感谢所有贡献者的支持！🙏

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 许可证

Ultralytics 提供两种许可证选项，以适配不同的使用场景：

- **AGPL-3.0 License**：这是一个经 [OSI 批准](https://opensource.org/license/agpl-3.0)的开源许可证，适用于学术研究、个人项目和实验用途。它鼓励开放协作与知识共享。详情请参阅 [LICENSE](https://github.com/ultralytics/yolo-ios-app/blob/main/LICENSE) 文件以及完整的 [AGPL-3.0 许可证文本](https://www.gnu.org/licenses/agpl-3.0.en.html)。
- **Enterprise License**：面向商业应用场景，允许将 Ultralytics 软件和 AI 模型集成到商业产品与服务中，而无需遵循 AGPL-3.0 的开源要求。如果你的场景涉及商业用途，请通过 [Ultralytics Licensing](https://www.ultralytics.com/license) 与我们联系。

## 🤝 联系我们

- 如需提交与该 iOS 项目相关的 bug 报告或功能请求，请使用 [GitHub Issues](https://github.com/ultralytics/yolo-ios-app/issues)。
- 如需咨询、讨论或获取与 Ultralytics 技术相关的支持，欢迎加入我们的 [Discord](https://discord.com/invite/ultralytics) 社区。

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
