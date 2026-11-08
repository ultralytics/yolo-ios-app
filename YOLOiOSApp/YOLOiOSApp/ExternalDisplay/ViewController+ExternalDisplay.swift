// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// MARK: - OPTIONAL External Display Support
// Optional extension that adds external monitor/TV support to the main ViewController. Dormant unless a display is
// connected.
//
// Responsibilities:
// - Detect external display connection/disconnection
// - Hide controls that don't apply in external-display mode (switch camera, share)
// - Mirror model and threshold changes to the external scene
// - Stop the iPhone camera while the external display is active

import UIKit
import UltralyticsYOLO

// MARK: - External Display Support
extension ViewController {

  func hasExternalScreen() -> Bool {
    UIApplication.shared.connectedScenes.contains {
      $0.session.role == .windowExternalDisplayNonInteractive
    }
  }

  func setupExternalDisplayNotifications() {
    // iOS 27 offers an external display scene only to apps that register it as a scene accessory; earlier systems
    // connect the Info.plist external display role on their own. The API ships with the iOS 27 SDK (Swift 6.4).
    #if compiler(>=6.4)
      if #available(iOS 27.0, *) {
        _ = registerSceneAccessory(
          .externalNonInteractive(
            sceneConfiguration: UISceneConfiguration(
              name: "External Display Configuration",
              sessionRole: .windowExternalDisplayNonInteractive)))
      }
    #endif

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
      selector: #selector(notifyExternalDisplayOfCurrentModel),
      name: .externalDisplayReady,
      object: nil
    )
  }

  @objc func handleExternalDisplayConnected(_ notification: Notification) {
    DispatchQueue.main.async {
      self.yoloView.stop()
      self.yoloView.setInferenceFlag(ok: false)
      self.showExternalDisplayStatus()

      DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
        self.view.setNeedsLayout()
        self.view.layoutIfNeeded()
        [
          self.yoloView.sliderConf, self.yoloView.labelSliderConf,
          self.yoloView.sliderIoU, self.yoloView.labelSliderIoU,
        ].forEach { $0?.isHidden = false }

        [
          self.yoloView.switchCameraButton,
          self.yoloView.shareButton,
          self.yoloView.infoButton,
        ].forEach { $0.isHidden = true }
        // Lens and torch controls derive their visibility from switchCameraButton during layout.
        self.yoloView.setNeedsLayout()
      }

      DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        self.sliderValueChanged(self.yoloView.sliderConf)
      }
    }
  }

  /// Sends the current model to the external display, which loads it into its own YOLOView.
  @objc func notifyExternalDisplayOfCurrentModel() {
    guard let entry = currentModels.first(where: { $0.identifier == currentModelIdentifier }),
      let task = appTasks.first(where: { $0.name == currentTask })
    else { return }
    let url =
      entry.installedURL
      ?? (entry.isLocalBundle
        ? Bundle.main.url(forResource: task.folder, withExtension: nil)?
          .appendingPathComponent(entry.identifier)
        : ModelCacheManager.shared.modelURL(for: entry.identifier))
    guard let url, FileManager.default.fileExists(atPath: url.path) else { return }
    NotificationCenter.default.post(
      name: .modelDidChange, object: nil,
      userInfo: ["task": task.yoloTask, "modelPath": url.path, "displayName": currentModelName])
  }

  @objc func handleExternalDisplayDisconnected(_ notification: Notification) {
    DispatchQueue.main.async {
      self.yoloView.isHidden = false
      self.hideExternalDisplayStatus()

      [
        self.yoloView.switchCameraButton,
        self.yoloView.shareButton,
        self.yoloView.infoButton,
      ].forEach { $0.isHidden = false }
      // Lens and torch controls derive their visibility from switchCameraButton during layout.
      self.yoloView.setNeedsLayout()

      self.yoloView.resume()
      self.yoloView.setInferenceFlag(ok: true)

      if let currentEntry = self.currentLoadingEntry {
        self.loadModel(entry: currentEntry, forTask: self.currentTask)
      } else if !self.currentModels.isEmpty {
        self.loadModel(
          entry: self.currentModels.first { $0.identifier == self.currentModelIdentifier }
            ?? self.currentModels[0],
          forTask: self.currentTask)
      }
    }
  }

  func showExternalDisplayStatus() {
    let statusLabel = UILabel()
    statusLabel.text = "📱 Camera is shown on external display"
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
