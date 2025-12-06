// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// MARK: - OPTIONAL External Display Support
// This extension provides optional external display functionality for the YOLO iOS app.
// It enhances the user experience when connected to an external monitor or TV but is
// NOT required for the core app functionality. The features remain dormant until
// an external display is connected.
//
// Features handled in this extension:
// - External display connection/disconnection detection
// - UI adjustments for external display mode:
//   * Hide switch camera and share buttons (not supported in external display mode)
//   * Adjust model dropdown positioning to prevent overlap
//   * Force landscape orientation for better external display experience
// - Model and threshold synchronization with external display
// - Camera session management (stop iPhone camera when external display is active)

import UIKit
import YOLO

// MARK: - External Display Support
extension ViewController {

  // Associated object key for tracking external display state
  private struct AssociatedKeys {
    static var isExternalDisplayConnectedKey: UInt8 = 0
  }

  private var isExternalDisplayConnected: Bool {
    get {
      return objc_getAssociatedObject(self, &AssociatedKeys.isExternalDisplayConnectedKey) as? Bool
        ?? false
    }
    set {
      objc_setAssociatedObject(
        self, &AssociatedKeys.isExternalDisplayConnectedKey, newValue,
        .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
  }
  
  private func hasExternalDisplayConnected() -> Bool {
    if #available(iOS 16.0, *) {
      let externalScenes = UIApplication.shared.openSessions
        .compactMap { $0.scene as? UIWindowScene }
        .filter { $0.screen != UIScreen.main }
      return !externalScenes.isEmpty
    } else {
      return UIScreen.screens.count > 1
    }
  }

  func setupExternalDisplayNotifications() {
    print("Setting up external display notifications")

    // Check if external display is already connected at startup
    if hasExternalDisplayConnected() {
      print("External display already connected at startup - triggering connection handler")
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        self.handleExternalDisplayConnected(Notification(name: .externalDisplayConnected))
      }
    }

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

    // Listen for detection count updates from external display
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleDetectionCountUpdate(_:)),
      name: .detectionCountDidUpdate,
      object: nil
    )
  }

  @objc func handleExternalDisplayConnected(_ notification: Notification) {
    DispatchQueue.main.async {
      self.isExternalDisplayConnected = true
      self.yoloView.stop()
      self.yoloView.setInferenceFlag(ok: false)
      self.showExternalDisplayStatus()

      self.requestLandscapeOrientation()

      DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
        self.adjustLayoutForExternalDisplayIfNeeded()

        self.view.setNeedsLayout()
        self.view.layoutIfNeeded()
        [
          self.yoloView.sliderConf, self.yoloView.labelSliderConf,
          self.yoloView.sliderIoU, self.yoloView.labelSliderIoU,
          self.yoloView.sliderNumItems, self.yoloView.labelSliderNumItems,
          self.yoloView.playButton, self.yoloView.pauseButton,
          self.modelSegmentedControl,
        ].forEach { $0?.isHidden = false }

        self.customModelButton?.isHidden = self.currentModels.isEmpty

        [
          self.yoloView.switchCameraButton,
          self.yoloView.shareButton,
        ].forEach { $0.isHidden = true }
        self.yoloView.labelSliderNumItems.text =
          "0 items (max \(Int(self.yoloView.sliderNumItems.value)))"

        self.yoloView.sliderNumItems.addTarget(
          self,
          action: #selector(self.updateNumItemsLabelForExternalDisplay),
          for: .valueChanged
        )
        self.modelSegmentedControl.setNeedsLayout()
        self.modelSegmentedControl.layoutIfNeeded()
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

  @objc private func updateNumItemsLabelForExternalDisplay() {
    if isExternalDisplayConnected {
      let maxValue = Int(yoloView.sliderNumItems.value)
      let currentText = yoloView.labelSliderNumItems.text ?? ""
      let currentCount = Int(currentText.split(separator: " ").first ?? "0") ?? 0
      yoloView.labelSliderNumItems.text = "\(currentCount) items (max \(maxValue))"
    }
  }

  @objc private func handleDetectionCountUpdate(_ notification: Notification) {
    guard isExternalDisplayConnected,
      let count = notification.userInfo?["count"] as? Int
    else { return }

    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      let maxValue = Int(self.yoloView.sliderNumItems.value)
      self.yoloView.labelSliderNumItems.text = "\(count) items (max \(maxValue))"
    }
  }

  func notifyExternalDisplayOfCurrentModel() {
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
      self.isExternalDisplayConnected = false
      self.yoloView.isHidden = false
      self.hideExternalDisplayStatus()

      self.modelSegmentedControl.isHidden = false
      [
        self.yoloView.switchCameraButton,
        self.yoloView.shareButton,
      ].forEach { $0.isHidden = false }

      self.yoloView.sliderNumItems.removeTarget(
        self,
        action: #selector(self.updateNumItemsLabelForExternalDisplay),
        for: .valueChanged
      )
      self.yoloView.sliderChanged(self.yoloView.sliderNumItems)

      self.yoloView.resume()
      self.yoloView.setInferenceFlag(ok: true)

      if let currentEntry = self.currentLoadingEntry {
        self.loadModel(entry: currentEntry, forTask: self.currentTask)
      } else if !self.currentModels.isEmpty {
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

  // MARK: - Layout Adjustments for External Display

  func adjustLayoutForExternalDisplayIfNeeded() {
    let hasExternalDisplay = hasExternalDisplayConnected() || SceneDelegate.hasExternalDisplay

    guard hasExternalDisplay else { return }

    // Layout adjustments for external display mode if needed
    // The segmented control and custom button are already properly positioned
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
    let hasExternalDisplay = hasExternalDisplayConnected()

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
      statusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 60),
    ])
  }

  func hideExternalDisplayStatus() {
    view.subviews.first(where: { $0.tag == 9999 })?.removeFromSuperview()
  }
}
