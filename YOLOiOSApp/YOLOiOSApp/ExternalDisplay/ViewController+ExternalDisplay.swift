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

  func hasExternalScreen() -> Bool {
    UIApplication.shared.connectedScenes
      .compactMap { $0 as? UIWindowScene }
      .contains { $0.screen != UIScreen.main }
  }

  func setupExternalDisplayNotifications() {
    if hasExternalScreen() {
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
        guard let self = self else { return }
        self.handleExternalDisplayConnected(Notification(name: .externalDisplayConnected))
      }
    }

    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayConnected(_:)),
      name: .externalDisplayConnected,
      object: nil
    )
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayDisconnected(_:)),
      name: .externalDisplayDisconnected,
      object: nil
    )
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

      self.requestLandscapeOrientation()

      DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
        self.view.setNeedsLayout()
        self.view.layoutIfNeeded()
        [
          self.yoloView.sliderConf, self.yoloView.labelSliderConf,
          self.yoloView.sliderIoU, self.yoloView.labelSliderIoU,
          self.yoloView.playButton, self.yoloView.pauseButton,
          self.modelSegmentedControl,
        ].forEach { $0?.isHidden = false }

        [
          self.yoloView.switchCameraButton,
          self.yoloView.shareButton,
          self.yoloView.infoButton,
        ].forEach { $0.isHidden = true }
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
    windowScene.requestGeometryUpdate(
      .iOS(interfaceOrientations: [.landscapeLeft, .landscapeRight]))
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
      self.yoloView.isHidden = false
      self.hideExternalDisplayStatus()

      self.modelSegmentedControl.isHidden = false
      [
        self.yoloView.switchCameraButton,
        self.yoloView.shareButton,
        self.yoloView.infoButton,
      ].forEach { $0.isHidden = false }

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
    windowScene.requestGeometryUpdate(
      .iOS(interfaceOrientations: [.portrait, .landscapeLeft, .landscapeRight]))
    setNeedsUpdateOfSupportedInterfaceOrientations()

    if UIDevice.current.orientation.isLandscape {
      UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
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
    guard hasExternalScreen() else { return }
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
      guard let self = self else { return }
      self.handleExternalDisplayReady(Notification(name: .externalDisplayReady))
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
