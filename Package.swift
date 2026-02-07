// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "YOLO",
  platforms: [.iOS(.v16)],
  products: [
    .library(name: "YOLOCore", targets: ["YOLOCore"]),
    .library(name: "YOLOUI", targets: ["YOLOUI"]),
  ],
  dependencies: [
    .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.19")
  ],
  targets: [
    .target(name: "YOLOCore", dependencies: ["ZIPFoundation"]),
    .target(name: "YOLOUI", dependencies: ["YOLOCore"]),
    .testTarget(
      name: "YOLOTests",
      dependencies: ["YOLOCore", "YOLOUI"],
      resources: [.process("Resources")]
    ),
  ]
)
