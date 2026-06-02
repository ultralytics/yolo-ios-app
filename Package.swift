// swift-tools-version: 5.10
// WARNING: <=5.10 required for GitHub Actions CI
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "UltralyticsYOLO",
  platforms: [
    .iOS(.v13)
  ],
  products: [
    // Products define the executables and libraries a package produces, making them visible to other packages.
    .library(
      name: "UltralyticsYOLO",
      targets: ["UltralyticsYOLO"])
  ],
  targets: [
    // Targets are the basic building blocks of a package, defining a module or a test suite.
    // Targets can depend on other targets in this package and products from dependencies.
    .target(
      name: "UltralyticsYOLO",
      exclude: ["README.md"]),
    .testTarget(
      name: "YOLOTests",
      dependencies: ["UltralyticsYOLO"],
      exclude: ["README.md"],
      resources: [
        .copy("Resources/yolo26n.mlpackage"),
        .copy("Resources/yolo26n-cls.mlpackage"),
        .copy("Resources/yolo26n-obb.mlpackage"),
        .copy("Resources/yolo26n-pose.mlpackage"),
        .copy("Resources/yolo26n-seg.mlpackage"),
        .copy("Resources/yolo26n-sem.mlpackage"),
      ]
    ),
  ]
)
