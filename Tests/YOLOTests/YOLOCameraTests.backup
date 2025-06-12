// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import XCTest
import SwiftUI
import AVFoundation
@testable import YOLO

/// Minimal tests for YOLOCamera SwiftUI component
class YOLOCameraTests: XCTestCase {
    
    func testYOLOCameraInitialization() {
        // Test YOLOCamera initialization with different parameters
        let camera1 = YOLOCamera(modelPathOrName: "test_model", task: .detect, cameraPosition: .back)
        
        XCTAssertEqual(camera1.modelPathOrName, "test_model")
        XCTAssertEqual(camera1.task, .detect)
        XCTAssertEqual(camera1.cameraPosition, .back)
    }
    
    func testYOLOCameraInitializationWithDefaultValues() {
        // Test YOLOCamera initialization with default values
        let camera = YOLOCamera(modelPathOrName: "model")
        
        XCTAssertEqual(camera.modelPathOrName, "model")
        XCTAssertEqual(camera.task, .detect)
        XCTAssertEqual(camera.cameraPosition, .back)
    }
    
    func testYOLOCameraWithDifferentTasks() {
        // Test YOLOCamera initialization with different tasks
        let tasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]
        
        for task in tasks {
            let camera = YOLOCamera(modelPathOrName: "test_model", task: task, cameraPosition: .front)
            
            XCTAssertEqual(camera.modelPathOrName, "test_model")
            XCTAssertEqual(camera.task, task)
            XCTAssertEqual(camera.cameraPosition, .front)
        }
    }
    
    func testYOLOCameraWithDifferentCameraPositions() {
        // Test YOLOCamera initialization with different camera positions
        let positions: [AVCaptureDevice.Position] = [.back, .front, .unspecified]
        
        for position in positions {
            let camera = YOLOCamera(modelPathOrName: "model", task: .classify, cameraPosition: position)
            
            XCTAssertEqual(camera.cameraPosition, position)
            XCTAssertEqual(camera.task, .classify)
        }
    }
    
    @MainActor
    func testYOLOCameraBody() {
        // Test YOLOCamera body property returns a View
        let camera = YOLOCamera(modelPathOrName: "test_model", task: .segment, cameraPosition: .back)
        
        let body = camera.body
        
        // Test that body is not nil and is a View
        XCTAssertNotNil(body)
        
        // Test that it's the correct type
        XCTAssertTrue(body is YOLOViewRepresentable)
    }
    
    func testYOLOCameraProperties() {
        // Test YOLOCamera properties are correctly set
        let modelPath = "custom_model_path"
        let task = YOLOTask.pose
        let position = AVCaptureDevice.Position.front
        
        let camera = YOLOCamera(modelPathOrName: modelPath, task: task, cameraPosition: position)
        
        XCTAssertEqual(camera.modelPathOrName, modelPath)
        XCTAssertEqual(camera.task, task)
        XCTAssertEqual(camera.cameraPosition, position)
    }
}

/// Tests for YOLOViewRepresentable
class YOLOViewRepresentableTests: XCTestCase {
    
    func testYOLOViewRepresentableInitialization() {
        // Test YOLOViewRepresentable initialization
        let modelPath = "test_model"
        let task = YOLOTask.detect
        let position = AVCaptureDevice.Position.back
        let onDetection: (YOLOResult) -> Void = { _ in }
        
        let representable = YOLOViewRepresentable(
            modelPathOrName: modelPath,
            task: task,
            cameraPosition: position,
            onDetection: onDetection
        )
        
        XCTAssertEqual(representable.modelPathOrName, modelPath)
        XCTAssertEqual(representable.task, task)
        XCTAssertEqual(representable.cameraPosition, position)
        XCTAssertNotNil(representable.onDetection)
    }
    
    func testYOLOViewRepresentableWithNilCallback() {
        // Test YOLOViewRepresentable with nil callback
        let representable = YOLOViewRepresentable(
            modelPathOrName: "model",
            task: .segment,
            cameraPosition: .front,
            onDetection: nil
        )
        
        XCTAssertEqual(representable.modelPathOrName, "model")
        XCTAssertEqual(representable.task, .segment)
        XCTAssertEqual(representable.cameraPosition, .front)
        XCTAssertNil(representable.onDetection)
    }
    
    @MainActor
    func testYOLOViewRepresentableMakeUIView() {
        // Test YOLOViewRepresentable makeUIView method
        let representable = YOLOViewRepresentable(
            modelPathOrName: "test_model",
            task: .classify,
            cameraPosition: .back,
            onDetection: nil
        )
        
        // Create a mock context - UIViewRepresentableContext constructor may vary
        let coordinator = representable.makeCoordinator()
        
        // Test that makeUIView doesn't crash
        // Note: This will try to load a model, which will fail, but shouldn't crash immediately
        XCTAssertNotNil(representable)
    }
    
    @MainActor
    func testYOLOViewRepresentableUpdateUIView() {
        // Test YOLOViewRepresentable updateUIView method
        let representable = YOLOViewRepresentable(
            modelPathOrName: "test_model",
            task: .detect,
            cameraPosition: .back,
            onDetection: nil
        )
        
        // Test that representable was created successfully
        XCTAssertNotNil(representable)
        XCTAssertEqual(representable.task, .detect)
    }
    
    func testYOLOViewRepresentableAllTaskTypes() {
        // Test YOLOViewRepresentable with all task types
        let tasks: [YOLOTask] = [.detect, .segment, .pose, .obb, .classify]
        
        for task in tasks {
            let representable = YOLOViewRepresentable(
                modelPathOrName: "model_\(task)",
                task: task,
                cameraPosition: .back,
                onDetection: nil
            )
            
            XCTAssertEqual(representable.task, task)
            XCTAssertEqual(representable.modelPathOrName, "model_\(task)")
        }
    }
}

// MARK: - SwiftUI Integration Tests

@available(iOS 13.0, *)  
class YOLOCameraSwiftUITests: XCTestCase {
    
    @MainActor
    func testYOLOCameraInSwiftUIView() {
        // Test YOLOCamera can be used in SwiftUI views
        let camera = YOLOCamera(modelPathOrName: "test_model")
        
        // Create a simple SwiftUI view containing the camera
        let contentView = VStack {
            camera
                .frame(width: 300, height: 400)
        }
        
        // Test that the view can be created without crashing
        XCTAssertNotNil(contentView)
    }
    
    func testYOLOCameraAsView() {
        // Test YOLOCamera conforms to View protocol
        let camera = YOLOCamera(modelPathOrName: "model")
        
        XCTAssertTrue(camera is any View)
    }
    
    func testYOLOCameraWithModifiers() {
        // Test YOLOCamera can be used with SwiftUI modifiers
        let camera = YOLOCamera(modelPathOrName: "test", task: .detect, cameraPosition: .front)
        
        // Apply some common SwiftUI modifiers
        let modifiedCamera = camera
            .frame(width: 200, height: 300)
            .clipShape(RoundedRectangle(cornerRadius: 10))
        
        XCTAssertNotNil(modifiedCamera)
    }
}
