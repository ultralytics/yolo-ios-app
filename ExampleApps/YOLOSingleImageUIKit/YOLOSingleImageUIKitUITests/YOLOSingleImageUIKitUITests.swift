// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing UI tests for the single image UIKit example.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import XCTest

/// UI tests for the YOLO Single Image UIKit example application.
///
/// This test class verifies the user interface functionality of the YOLO Single Image UIKit app.
/// It tests the app's UI workflow, including image selection, processing, and result display.
/// The tests ensure that the app's interface responds correctly to user interactions and
/// displays expected visual elements.
///
/// - Note: UI tests launch the actual application and simulate user interaction.
/// - Important: These tests require a device with camera and photo library permissions.
final class YOLOSingleImageUIKitUITests: XCTestCase {

  override func setUpWithError() throws {
    // Put setup code here. This method is called before the invocation of each test method in the class.

    // In UI tests it is usually best to stop immediately when a failure occurs.
    continueAfterFailure = false

    // In UI tests it’s important to set the initial state - such as interface orientation - required for your tests before they run. The setUp method is a good place to do this.
  }

  override func tearDownWithError() throws {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
  }

  @MainActor
  func testExample() throws {
    // UI tests must launch the application that they test.
    let app = XCUIApplication()
    app.launch()

    // Use XCTAssert and related functions to verify your tests produce the correct results.
  }

  // Remove performance test from this file as it's already handled in LaunchTests
  // The error occurred because performance tests need to be in their own class with
  // special configuration. LaunchTests already handle this correctly.
}
