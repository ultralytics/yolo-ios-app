// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest

@testable import YOLO

/// Minimal tests for YOLOTask enum
class YOLOTaskTests: XCTestCase {

  func testAllTaskTypes() {
    // Test all YOLOTask enum cases exist and are distinct
    let tasks: [YOLOTask] = [.detect, .segment, .semantic, .pose, .obb, .classify]

    XCTAssertEqual(tasks.count, 6)

    // Test each task type
    XCTAssertNotEqual(YOLOTask.detect, YOLOTask.segment)
    XCTAssertNotEqual(YOLOTask.detect, YOLOTask.pose)
    XCTAssertNotEqual(YOLOTask.detect, YOLOTask.obb)
    XCTAssertNotEqual(YOLOTask.detect, YOLOTask.classify)
    XCTAssertNotEqual(YOLOTask.segment, YOLOTask.pose)
    XCTAssertNotEqual(YOLOTask.segment, YOLOTask.semantic)
  }

  func testTaskEquality() {
    // Test YOLOTask equality
    XCTAssertEqual(YOLOTask.detect, YOLOTask.detect)
    XCTAssertEqual(YOLOTask.segment, YOLOTask.segment)
    XCTAssertEqual(YOLOTask.semantic, YOLOTask.semantic)
    XCTAssertEqual(YOLOTask.pose, YOLOTask.pose)
    XCTAssertEqual(YOLOTask.obb, YOLOTask.obb)
    XCTAssertEqual(YOLOTask.classify, YOLOTask.classify)
  }

  func testTaskSwitchStatement() {
    // Test YOLOTask can be used in switch statements
    func taskDescription(_ task: YOLOTask) -> String {
      switch task {
      case .detect: return "detection"
      case .segment: return "segmentation"
      case .semantic: return "semantic"
      case .pose: return "pose"
      case .obb: return "obb"
      case .classify: return "classification"
      }
    }

    XCTAssertEqual(taskDescription(.detect), "detection")
    XCTAssertEqual(taskDescription(.segment), "segmentation")
    XCTAssertEqual(taskDescription(.semantic), "semantic")
    XCTAssertEqual(taskDescription(.pose), "pose")
    XCTAssertEqual(taskDescription(.obb), "obb")
    XCTAssertEqual(taskDescription(.classify), "classification")
  }

  func testTaskInArray() {
    // Test YOLOTask can be stored in arrays and collections
    let detectionTasks: [YOLOTask] = [.detect, .obb]
    let segmentationTasks: [YOLOTask] = [.segment, .semantic]
    let humanTasks: [YOLOTask] = [.pose]
    let classificationTasks: [YOLOTask] = [.classify]

    XCTAssertTrue(detectionTasks.contains(.detect))
    XCTAssertTrue(detectionTasks.contains(.obb))
    XCTAssertFalse(detectionTasks.contains(.segment))

    XCTAssertTrue(segmentationTasks.contains(.segment))
    XCTAssertTrue(segmentationTasks.contains(.semantic))
    XCTAssertTrue(humanTasks.contains(.pose))
    XCTAssertTrue(classificationTasks.contains(.classify))
  }
}
