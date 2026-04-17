// swift-tools-version: 5.10
// WARNING: <=5.10 requires for GitHub Actions CI
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "YOLO",
  platforms: [
    .iOS(.v16)
  ],
  products: [
    // Products define the executables and libraries a package produces, making them visible to other packages.
    .library(
      name: "YOLO",
      targets: ["YOLO"])
  ],
  dependencies: [
    .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.20")
  ],
  targets: [
    // Targets are the basic building blocks of a package, defining a module or a test suite.
    // Targets can depend on other targets in this package and products from dependencies.
    .target(
      name: "YOLO",
      dependencies: ["ZIPFoundation"],
      exclude: ["README.md"]),
    .testTarget(
      name: "YOLOTests",
      dependencies: ["YOLO"],
      exclude: ["README.md"],
      resources: [
        .copy("Resources/yolo26n.mlpackage"),
        .copy("Resources/yolo26n-cls.mlpackage"),
        .copy("Resources/yolo26n-obb.mlpackage"),
        .copy("Resources/yolo26n-pose.mlpackage"),
        .copy("Resources/yolo26n-seg.mlpackage"),
      ]
    ),
  ]
)
