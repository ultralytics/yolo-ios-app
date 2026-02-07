// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest

@testable import YOLOCore

/// Tests for YOLOTask enum
class YOLOTaskTests: XCTestCase {

  func testAllTaskTypes() {
    let tasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]
    XCTAssertEqual(tasks.count, 5)

    XCTAssertNotEqual(YOLOTask.detect, YOLOTask.segment)
    XCTAssertNotEqual(YOLOTask.detect, YOLOTask.pose)
    XCTAssertNotEqual(YOLOTask.detect, YOLOTask.obb)
    XCTAssertNotEqual(YOLOTask.detect, YOLOTask.classify)
    XCTAssertNotEqual(YOLOTask.segment, YOLOTask.pose)
  }

  func testTaskEquality() {
    XCTAssertEqual(YOLOTask.detect, YOLOTask.detect)
    XCTAssertEqual(YOLOTask.segment, YOLOTask.segment)
    XCTAssertEqual(YOLOTask.pose, YOLOTask.pose)
    XCTAssertEqual(YOLOTask.obb, YOLOTask.obb)
    XCTAssertEqual(YOLOTask.classify, YOLOTask.classify)
  }

  func testTaskSwitchStatement() {
    func taskDescription(_ task: YOLOTask) -> String {
      switch task {
      case .detect: return "detection"
      case .segment: return "segmentation"
      case .pose: return "pose"
      case .obb: return "obb"
      case .classify: return "classification"
      }
    }

    XCTAssertEqual(taskDescription(.detect), "detection")
    XCTAssertEqual(taskDescription(.segment), "segmentation")
    XCTAssertEqual(taskDescription(.pose), "pose")
    XCTAssertEqual(taskDescription(.obb), "obb")
    XCTAssertEqual(taskDescription(.classify), "classification")
  }

  func testTaskInArray() {
    let detectionTasks: [YOLOTask] = [.detect, .obb]
    XCTAssertTrue(detectionTasks.contains(.detect))
    XCTAssertTrue(detectionTasks.contains(.obb))
    XCTAssertFalse(detectionTasks.contains(.segment))
  }
}
