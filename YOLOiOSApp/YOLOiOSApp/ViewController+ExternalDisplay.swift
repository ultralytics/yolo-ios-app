// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit
import YOLO

// MARK: - External Display Support
extension ViewController {
  
  func setupExternalDisplayNotifications() {
    // Listen for external display connection
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayConnected(_:)),
      name: .externalDisplayConnected,
      object: nil
    )
    
    // Listen for external display disconnection
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayDisconnected(_:)),
      name: .externalDisplayDisconnected,
      object: nil
    )
    
    // Listen for when external display is ready
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayReady(_:)),
      name: .externalDisplayReady,
      object: nil
    )
  }
  
  @objc func handleExternalDisplayConnected(_ notification: Notification) {
    print("External display connected")
    
    // Stop camera but keep UI visible on iPhone when external display is connected
    DispatchQueue.main.async {
      // Stop the video capture to release camera for external display
      self.yoloView.stop()
      
      // Disable inference on main display
      self.yoloView.setInferenceFlag(ok: false)
      
      // Don't hide the YOLOView - just keep it visible with controls
      // The stop() method should have already stopped the camera feed
      
      self.showExternalDisplayStatus()
      
      // Make sure sliders remain visible and functional
      self.yoloView.sliderConf.isHidden = false
      self.yoloView.labelSliderConf.isHidden = false
      self.yoloView.sliderIoU.isHidden = false
      self.yoloView.labelSliderIoU.isHidden = false
      self.yoloView.sliderNumItems.isHidden = false
      self.yoloView.labelSliderNumItems.isHidden = false
      
      // Also ensure buttons are visible
      self.yoloView.playButton.isHidden = false
      self.yoloView.pauseButton.isHidden = false
      self.yoloView.switchCameraButton.isHidden = false
      
      // Keep model table view visible for model selection
      self.modelTableView.isHidden = false
      self.tableViewBGView.isHidden = false
      
      print("🔴 Stopped main YOLOView camera but kept controls visible")
      
      // Force orientation update when external display connects
      if let windowScene = self.view.window?.windowScene {
        if #available(iOS 16.0, *) {
          windowScene.requestGeometryUpdate(.iOS(interfaceOrientations: [.landscapeLeft, .landscapeRight]))
        } else {
          UIViewController.attemptRotationToDeviceOrientation()
        }
      }
      
      // Wait a bit before sending model info to ensure external display is ready
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        // Send current model to external display
        if let entry = self.currentLoadingEntry ?? self.currentModels.first {
          self.notifyExternalDisplayOfCurrentModel()
        }
        
        // Send current threshold values
        self.sliderValueChanged(self.yoloView.sliderConf)
        
        // Send current task
        NotificationCenter.default.post(
          name: .taskDidChange,
          object: nil,
          userInfo: ["task": self.currentTask]
        )
      }
    }
  }
  
  func notifyExternalDisplayOfCurrentModel() {
    // Get current model info and send to external display
    let yoloTask = convertTaskNameToYOLOTask(currentTask)
    
    var fullModelPath = currentModelName
    if let entry = currentLoadingEntry ?? currentModels.first(where: { processString($0.displayName) == currentModelName }),
       entry.isLocalBundle,
       let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
       let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil) {
        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        fullModelPath = modelURL.path
    }
    
    ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
  }
  
  @objc func handleExternalDisplayDisconnected(_ notification: Notification) {
    print("External display disconnected")
    
    // Show and restart YOLOView on main display when external display is disconnected
    DispatchQueue.main.async {
      self.yoloView.isHidden = false
      self.hideExternalDisplayStatus()
      
      // Show model table view again
      self.modelTableView.isHidden = false
      self.tableViewBGView.isHidden = false
      
      // Restart the main YOLOView video capture
      self.yoloView.resume()
      print("🟢 Restarted main YOLOView video capture")
      
      // Re-enable inference flag to restart detection
      self.yoloView.setInferenceFlag(ok: true)
      
      // If no model is currently selected, auto-select the first one
      if self.selectedIndexPath == nil && !self.currentModels.isEmpty {
        print("🟢 Auto-selecting first model after external display disconnection")
        let firstIndex = IndexPath(row: 0, section: 0)
        self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
        self.selectedIndexPath = firstIndex
        let firstModel = self.currentModels[0]
        self.loadModel(entry: firstModel, forTask: self.currentTask)
      }
      
      // Force orientation update when external display disconnects
      if let windowScene = self.view.window?.windowScene {
        if #available(iOS 16.0, *) {
          windowScene.requestGeometryUpdate(.iOS(interfaceOrientations: [.portrait, .landscapeLeft, .landscapeRight]))
          // Force immediate orientation update
          self.setNeedsUpdateOfSupportedInterfaceOrientations()
          
          // If currently in landscape, try to rotate to portrait
          if UIDevice.current.orientation.isLandscape {
            let value = UIInterfaceOrientation.portrait.rawValue
            UIDevice.current.setValue(value, forKey: "orientation")
          }
        } else {
          UIViewController.attemptRotationToDeviceOrientation()
          
          // For older iOS versions, force rotation
          if UIDevice.current.orientation.isLandscape {
            let value = UIInterfaceOrientation.portrait.rawValue
            UIDevice.current.setValue(value, forKey: "orientation")
          }
        }
      }
    }
  }
  
  @objc func handleExternalDisplayReady(_ notification: Notification) {
    print("External display ready to receive content")
    print("  - currentTask: \(currentTask)")
    print("  - currentModelName: \(currentModelName)")
    print("  - currentModels count: \(currentModels.count)")
    
    // Only send model if we have one loaded
    guard !currentTask.isEmpty && !currentModels.isEmpty else {
        print("  - No task or models loaded yet, skipping notification")
        return
    }
    
    // Share current task/model info
    let yoloTask = convertTaskNameToYOLOTask(currentTask)
    
    // Get the full model path for external display
    var fullModelPath = ""
    
    // Use the first model if no specific model is selected yet
    if let currentEntry = currentModels.first(where: { processString($0.displayName) == currentModelName }) ?? currentModels.first {
        print("  - Found model entry: \(currentEntry.displayName)")
        
        if currentEntry.isLocalBundle,
           let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
           let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil) {
            let modelURL = folderPathURL.appendingPathComponent(currentEntry.identifier)
            fullModelPath = modelURL.path
            print("  - Full model path: \(fullModelPath)")
        }
    } else {
        print("  - No model entry found!")
        return
    }
    
    // Only notify if we have a valid path
    guard !fullModelPath.isEmpty else {
        print("  - Empty model path, skipping notification")
        return
    }
    
    ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
  }
  
  func checkAndNotifyExternalDisplayIfReady() {
    // Check if external display is connected and ready
    let hasExternalDisplay = UIApplication.shared.connectedScenes
      .compactMap({ $0 as? UIWindowScene })
      .contains(where: { $0.screen != UIScreen.main })
    
    if hasExternalDisplay {
      print("External display detected, sending current model")
      // Wait a bit to ensure external display is fully initialized
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        self.handleExternalDisplayReady(Notification(name: .externalDisplayReady))
      }
    }
  }
  
  func checkForExternalDisplays() {
    print("Checking for external displays...")
    print("Number of screens: \(UIScreen.screens.count)")
    
    if UIScreen.screens.count > 1 {
      print("External display detected!")
      for (index, screen) in UIScreen.screens.enumerated() {
        print("Screen \(index): \(screen)")
        print("  Bounds: \(screen.bounds)")
        print("  Scale: \(screen.scale)")
        print("  Available modes: \(screen.availableModes.count)")
        for mode in screen.availableModes {
          print("    Mode: \(mode.size)")
        }
      }
      
      // Check if external display scene is active
      if let windowScene = UIApplication.shared.connectedScenes
        .compactMap({ $0 as? UIWindowScene })
        .first(where: { $0.screen != UIScreen.main }) {
        print("External display window scene found: \(windowScene)")
      } else {
        print("No external display window scene found")
      }
    } else {
      print("No external display connected")
    }
  }
  
  func showExternalDisplayStatus() {
    // Create a label to show external display status
    let statusLabel = UILabel()
    statusLabel.text = "📱 Camera is shown on external display\n🔄 Please use landscape orientation"
    statusLabel.textColor = .white
    statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.8)
    statusLabel.textAlignment = .center
    statusLabel.font = .systemFont(ofSize: 18, weight: .medium)
    statusLabel.numberOfLines = 0  // Allow unlimited lines
    statusLabel.adjustsFontSizeToFitWidth = true
    statusLabel.minimumScaleFactor = 0.8
    statusLabel.layer.cornerRadius = 10
    statusLabel.layer.masksToBounds = true
    statusLabel.tag = 9999 // Tag for later removal
    
    view.addSubview(statusLabel)
    
    statusLabel.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      statusLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
      statusLabel.widthAnchor.constraint(equalToConstant: 280),
      statusLabel.heightAnchor.constraint(equalToConstant: 140)
    ])
  }
  
  func hideExternalDisplayStatus() {
    // Remove the status label
    view.subviews.first(where: { $0.tag == 9999 })?.removeFromSuperview()
  }
}