// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// MARK: - OPTIONAL External Display Support
// This extension provides optional external display functionality for the YOLO iOS app.
// It enhances the user experience when connected to an external monitor or TV but is
// NOT required for the core app functionality. The features remain dormant until
// an external display is connected.

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
    DispatchQueue.main.async {
      self.yoloView.stop()
      self.yoloView.setInferenceFlag(ok: false)
      self.showExternalDisplayStatus()

      // First request orientation change
      self.requestLandscapeOrientation()
      
      // Delay UI updates to allow orientation change to complete
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
        // Force layout update after orientation change
        self.view.setNeedsLayout()
        self.view.layoutIfNeeded()
        
        // Keep controls visible except switch camera and share buttons
        [
          self.yoloView.sliderConf, self.yoloView.labelSliderConf,
          self.yoloView.sliderIoU, self.yoloView.labelSliderIoU,
          self.yoloView.sliderNumItems, self.yoloView.labelSliderNumItems,
          self.yoloView.playButton, self.yoloView.pauseButton,
          self.modelTableView, self.tableViewBGView,
        ].forEach { $0.isHidden = false }
        
        // Hide switch camera and share buttons in external display mode
        [
          self.yoloView.switchCameraButton,
          self.yoloView.shareButton,
        ].forEach { $0.isHidden = true }
        
        // Update table view constraints after orientation change
        self.modelTableView.setNeedsLayout()
        self.modelTableView.layoutIfNeeded()
        self.tableViewBGView.setNeedsLayout()
        self.tableViewBGView.layoutIfNeeded()
      }

      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        if self.currentLoadingEntry != nil || !self.currentModels.isEmpty {
          self.notifyExternalDisplayOfCurrentModel()
        }
        self.sliderValueChanged(self.yoloView.sliderConf)
        NotificationCenter.default.post(
          name: .taskDidChange,
          object: nil,
          userInfo: ["task": self.currentTask]
        )
      }
    }
  }

  private func requestLandscapeOrientation() {
    guard let windowScene = view.window?.windowScene else { return }

    if #available(iOS 16.0, *) {
      windowScene.requestGeometryUpdate(
        .iOS(interfaceOrientations: [.landscapeLeft, .landscapeRight]))
    } else {
      UIViewController.attemptRotationToDeviceOrientation()
    }
  }

  func notifyExternalDisplayOfCurrentModel() {
    // Get current model info and send to external display
    let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect

    var fullModelPath = currentModelName
    if let entry = currentLoadingEntry
      ?? currentModels.first(where: { processString($0.displayName) == currentModelName }),
      entry.isLocalBundle,
      let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
      let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
    {
      let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
      fullModelPath = modelURL.path
    }

    ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
  }

  @objc func handleExternalDisplayDisconnected(_ notification: Notification) {
    DispatchQueue.main.async {
      self.yoloView.isHidden = false
      self.hideExternalDisplayStatus()

      self.modelTableView.isHidden = false
      self.tableViewBGView.isHidden = false
      
      // Show switch camera and share buttons again when returning to iPhone-only mode
      [
        self.yoloView.switchCameraButton,
        self.yoloView.shareButton,
      ].forEach { $0.isHidden = false }

      self.yoloView.resume()
      self.yoloView.setInferenceFlag(ok: true)

      if self.selectedIndexPath == nil && !self.currentModels.isEmpty {
        let firstIndex = IndexPath(row: 0, section: 0)
        self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
        self.selectedIndexPath = firstIndex
        self.loadModel(entry: self.currentModels[0], forTask: self.currentTask)
      }

      self.requestPortraitOrientation()
    }
  }

  private func requestPortraitOrientation() {
    guard let windowScene = view.window?.windowScene else { return }

    if #available(iOS 16.0, *) {
      windowScene.requestGeometryUpdate(
        .iOS(interfaceOrientations: [.portrait, .landscapeLeft, .landscapeRight]))
      setNeedsUpdateOfSupportedInterfaceOrientations()

      if UIDevice.current.orientation.isLandscape {
        UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
      }
    } else {
      UIViewController.attemptRotationToDeviceOrientation()

      if UIDevice.current.orientation.isLandscape {
        UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
      }
    }
  }

  @objc func handleExternalDisplayReady(_ notification: Notification) {
    guard !currentTask.isEmpty && !currentModels.isEmpty else { return }

    let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect

    let currentEntry =
      currentModels.first(where: { processString($0.displayName) == currentModelName })
      ?? currentModels.first
    guard let entry = currentEntry else { return }

    var fullModelPath = ""
    if entry.isLocalBundle,
      let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
      let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
    {
      fullModelPath = folderPathURL.appendingPathComponent(entry.identifier).path
    }

    guard !fullModelPath.isEmpty else { return }

    ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
  }

  func checkAndNotifyExternalDisplayIfReady() {
    let hasExternalDisplay = UIApplication.shared.connectedScenes
      .compactMap({ $0 as? UIWindowScene })
      .contains(where: { $0.screen != UIScreen.main })

    if hasExternalDisplay {
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        self.handleExternalDisplayReady(Notification(name: .externalDisplayReady))
      }
    }
  }

  func checkForExternalDisplays() {
    let hasExternalDisplay = UIScreen.screens.count > 1

    if hasExternalDisplay {
      _ = UIApplication.shared.connectedScenes
        .compactMap({ $0 as? UIWindowScene })
        .first(where: { $0.screen != UIScreen.main })
    }
  }

  func showExternalDisplayStatus() {
    let statusLabel = UILabel()
    statusLabel.text = "📱 Camera is shown on external display\n🔄 Please use landscape orientation"
    statusLabel.textColor = .white
    statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.8)
    statusLabel.textAlignment = .center
    statusLabel.font = .systemFont(ofSize: 18, weight: .medium)
    statusLabel.numberOfLines = 0
    statusLabel.adjustsFontSizeToFitWidth = true
    statusLabel.minimumScaleFactor = 0.8
    statusLabel.layer.cornerRadius = 10
    statusLabel.layer.masksToBounds = true
    statusLabel.tag = 9999

    view.addSubview(statusLabel)

    statusLabel.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      statusLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
      statusLabel.widthAnchor.constraint(equalToConstant: 280),
      statusLabel.heightAnchor.constraint(equalToConstant: 140),
    ])
  }

  func hideExternalDisplayStatus() {
    view.subviews.first(where: { $0.tag == 9999 })?.removeFromSuperview()
  }
}
