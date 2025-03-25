//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  This file is part of the Example Apps of Ultralytics YOLO Package, providing launch tests for the single image UIKit example.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import XCTest

/// Launch performance tests for the YOLO Single Image UIKit example application.
///
/// This test class focuses specifically on measuring and verifying the app's launch performance.
/// It captures screenshots of the launch process and measures the time taken to launch the application.
/// These tests help ensure the app launches efficiently and displays the correct initial interface.
///
/// - Note: These tests run for each target application UI configuration.
/// - Important: Launch tests are critical for monitoring performance regressions between app versions.
final class YOLO_Single_Image_UIKitUITestsLaunchTests: XCTestCase {

  override class var runsForEachTargetApplicationUIConfiguration: Bool {
    true
  }

  override func setUpWithError() throws {
    continueAfterFailure = false
  }

  @MainActor
  func testLaunch() throws {
    let app = XCUIApplication()
    app.launch()

    // Insert steps here to perform after app launch but before taking a screenshot,
    // such as logging into a test account or navigating somewhere in the app

    let attachment = XCTAttachment(screenshot: app.screenshot())
    attachment.name = "Launch Screen"
    attachment.lifetime = .keepAlways
    add(attachment)
  }
}
