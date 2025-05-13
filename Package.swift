// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

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
  targets: [
    // Targets are the basic building blocks of a package, defining a module or a test suite.
    // Targets can depend on other targets in this package and products from dependencies.
    .target(
      name: "YOLO"),
    .testTarget(
      name: "YOLOTests",
      dependencies: ["YOLO"],
      resources: [
        .process("Resources")
      ]
    ),
  ]
)
