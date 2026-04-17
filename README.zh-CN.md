<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 适用于 iOS 的 Ultralytics YOLO：App 与 Swift Package

[![Ultralytics Actions](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/format.yml)
[![CI](https://github.com/ultralytics/yolo-ios-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-ios-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-ios-app/branch/main/graph/badge.svg)](https://codecov.io/gh/ultralytics/yolo-ios-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

欢迎来到 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) iOS App 的 GitHub 仓库！📖 这个项目可将你的 iOS 设备变成强大的[实时推理](https://www.ultralytics.com/glossary/real-time-inference)工具，支持[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割、分类、姿态估计以及旋转框检测。该 SDK 同时支持传统的 [YOLO11](https://docs.ultralytics.com/models/yolo11/)（使用 Core ML NMS）和最新的 [YOLO26 模型](https://platform.ultralytics.com/ultralytics/yolo26)（无 NMS，使用 Swift 侧后处理）。你可以直接从 [App Store](https://apps.apple.com/us/app/idetection/id1452689527) 下载应用，也可以参考本指南，将 YOLO 能力集成到你自己的 Swift 应用中。

<div align="center">
  <br>
  <a href="https://apps.apple.com/us/app/idetection/id1452689527" target="_blank"><img width="100%" src="https://github.com/user-attachments/assets/d5dab2e7-f473-47ce-bc63-69bef89ba52a" alt="Ultralytics YOLO iOS App previews"></a>
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
  <a href="https://apps.apple.com/us/app/idetection/id1452689527" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/app-store.svg" width="15%" alt="Apple App store"></a>
</div>

## ✨ 为什么选择原生 YOLO iOS？

| 功能     | iOS |
| -------- | --- |
| 检测     | ✅  |
| 分类     | ✅  |
| 分割     | ✅  |
| 姿态估计 | ✅  |
| OBB 检测 | ✅  |

- 原生 iOS 性能 - 通过 Swift 与 Core ML 获得最高速度
- 针对 Apple Silicon 优化 - 充分利用 Neural Engine 与 GPU
- 实时检测 - 在最新款 iPhone 上可达 60+ FPS
- 低延迟 - 无框架额外开销，直接访问硬件能力
- iOS 优先设计 - 原生 UI/UX，遵循 Apple 设计规范
- Core ML 集成 - 使用 Apple 官方优化的机器学习框架

## 📂 仓库内容

此仓库为在 Apple 平台上运行 YOLO 模型提供了完整方案，帮助你构建强大的[边缘 AI](https://www.ultralytics.com/glossary/edge-ai)能力：

### [**Ultralytics YOLO iOS App（主应用）**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp)

这是主要的 iOS 应用，可通过设备相机或图片库轻松进行实时目标检测。你还可以通过简单拖放，将自己的 [Core ML](https://developer.apple.com/documentation/coreml) 模型导入应用中快速测试。

### [**Swift Package（YOLO 库）**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/YOLO)

这是一个面向 iOS 和 iPadOS 的轻量级 [Swift](https://developer.apple.com/swift/) package，用于简化 YOLO26 等 YOLO 模型在应用中的集成与使用。借助 [SwiftUI](https://developer.apple.com/xcode/swiftui/)，你可以用极少的代码轻松集成 YOLO：

```swift
// 对 UIImage 执行推理
let result = model(uiImage)
```

```swift
// 使用内置相机视图进行实时检测
var body: some View {
    YOLOCamera(
        modelPathOrName: "yolo26n-seg", // 指定模型名称或路径
        task: .segment,                // 定义任务（detect、segment、classify、pose）
        cameraPosition: .back          // 选择摄像头（后置或前置）
    )
    .ignoresSafeArea()
}
```

## 🛠️ 快速开始

如果你刚接触移动端 YOLO，或想快速测试自己的模型，建议先从主应用 YOLOiOSApp 开始。

- [**Ultralytics YOLO iOS App（主应用）**](https://github.com/ultralytics/yolo-ios-app/tree/main/YOLOiOSApp)：在 iOS 上体验 YOLO 检测的最简单方式。

如果你已经准备好将 YOLO 集成到自己的项目中，可以继续查看 Swift Package 和示例应用。

- [**Swift Package（YOLO 库）**](https://github.com/ultralytics/yolo-ios-app/tree/main/Sources/YOLO)：将 YOLO 能力集成到你的 Swift 应用中。
- [**示例应用**](https://github.com/ultralytics/yolo-ios-app/tree/main/ExampleApps)：查看基于 YOLO Swift Package 的实际实现示例。

## ✨ 核心亮点

- **实时推理**：使用优化后的 [Core ML 模型](https://docs.ultralytics.com/integrations/coreml/)，在 iPhone 和 iPad 上实现高速、高精度的目标检测，并可结合[模型量化](https://www.ultralytics.com/glossary/model-quantization)等技术进一步提升性能。
- **Apple 移动平台支持**：Swift Package 面向 iOS 和 iPadOS，并提供原生 Core ML 集成。
- **灵活任务支持**：支持[目标检测](https://docs.ultralytics.com/tasks/detect/)、[分割](https://docs.ultralytics.com/tasks/segment/)、[分类](https://docs.ultralytics.com/tasks/classify/)、[姿态估计](https://docs.ultralytics.com/tasks/pose/)以及[旋转框（OBB）检测](https://docs.ultralytics.com/tasks/obb/)。

## 🧪 测试流程

此仓库为 YOLO Swift Package 和示例应用都提供了较完整的[单元测试](https://en.wikipedia.org/wiki/Unit_testing)，以确保代码的可靠性与稳定性。

### 运行测试

测试依赖 Core ML 模型文件（`.mlpackage`），但由于文件体积较大，仓库中不包含这些模型。若要执行带模型校验的测试，请按以下步骤操作：

1. 在相关测试文件中将 `SKIP_MODEL_TESTS = false`。
2. 从 [Ultralytics 发布页](https://github.com/ultralytics/ultralytics/releases)下载所需模型，或通过 [Ultralytics Platform](https://platform.ultralytics.com) 训练你自己的模型。
3. 使用 [Ultralytics Python 库的导出功能](https://docs.ultralytics.com/modes/export/) 将模型转换为 Core ML 格式。
4. 将导出的 `.mlpackage` 文件添加到你的 [Xcode](https://developer.apple.com/xcode/) 项目中，并确保它们已加入对应的测试 target。
5. 通过 Xcode 的 Test Navigator（Cmd+U）运行测试。

如果你没有这些模型文件，也可以保持 `SKIP_MODEL_TESTS = true`。这样会跳过需要加载和运行模型的测试。

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

我们非常欢迎你为开源项目贡献力量！你的支持将帮助我们持续推动[人工智能（AI）](https://www.ultralytics.com/glossary/artificial-intelligence-ai)的发展边界。欢迎查阅[贡献指南](https://docs.ultralytics.com/help/contributing/)，也可以通过我们的[问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享你的反馈。感谢所有贡献者的支持！🙏

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
