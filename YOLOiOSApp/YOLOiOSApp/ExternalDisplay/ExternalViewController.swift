// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit
import UltralyticsYOLO

/// Shows the camera feed full screen on an external display with YOLOView's own model name and performance labels.
class ExternalViewController: UIViewController {

  private(set) var yoloView: YOLOView?

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .black

    let logoImageView = UIImageView(image: UIImage(named: "ultralytics_yolo_logotype"))
    logoImageView.contentMode = .scaleAspectFit
    logoImageView.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(logoImageView)
    NSLayoutConstraint.activate([
      logoImageView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
      logoImageView.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -20),
      logoImageView.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.2),
      logoImageView.heightAnchor.constraint(equalTo: logoImageView.widthAnchor, multiplier: 0.3),
    ])

    NotificationCenter.default.addObserver(
      self, selector: #selector(handleModelChange(_:)), name: .modelDidChange, object: nil)
    NotificationCenter.default.addObserver(
      self, selector: #selector(handleThresholdChange(_:)), name: .thresholdDidChange, object: nil)
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
    guard yoloView == nil else { return }

    // Created once the view has its final size so YOLOView scales its boxes for the large display. The main app
    // supplies the model via .modelDidChange.
    let yoloView = YOLOView(frame: view.bounds)
    yoloView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    view.insertSubview(yoloView, at: 0)
    self.yoloView = yoloView

    // YOLOView sizes its labels for a phone; scale them to the display height.
    let height = view.bounds.height
    yoloView.labelName.font = .systemFont(ofSize: height * 0.08, weight: .bold)
    yoloView.labelFPS.font = .systemFont(ofSize: height * 0.03, weight: .medium)
    yoloView.labelBreakdown.font = .systemFont(ofSize: height * 0.022)

    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
      self?.hideYOLOViewControls()
      NotificationCenter.default.post(name: .externalDisplayReady, object: nil)
    }
  }

  /// The external display is non-interactive, so YOLOView's touch controls stay on the iPhone.
  private func hideYOLOViewControls() {
    guard let yoloView = yoloView else { return }

    [
      yoloView.sliderConf, yoloView.labelSliderConf,
      yoloView.sliderIoU, yoloView.labelSliderIoU,
      yoloView.toolbar,
    ].forEach { $0.removeFromSuperview() }

    [
      yoloView.labelZoom, yoloView.activityIndicator,
      yoloView.playButton, yoloView.pauseButton,
      yoloView.switchCameraButton, yoloView.infoButton,
    ].forEach { $0.isHidden = true }
  }

  @objc private func handleModelChange(_ notification: Notification) {
    guard let userInfo = notification.userInfo,
      let task = userInfo["task"] as? YOLOTask,
      let modelPath = userInfo["modelPath"] as? String,
      let displayName = userInfo["displayName"] as? String
    else { return }

    yoloView?.setModel(modelPathOrName: modelPath, task: task) { [weak self] result in
      guard case .success = result else { return }
      // YOLOView names the model after its file, which for downloads is the cache name.
      DispatchQueue.main.async { self?.yoloView?.labelName.text = displayName }
    }
  }

  @objc private func handleThresholdChange(_ notification: Notification) {
    guard let userInfo = notification.userInfo,
      let conf = userInfo["conf"] as? Double,
      let iou = userInfo["iou"] as? Double,
      let maxItems = userInfo["maxItems"] as? Int
    else { return }

    yoloView?.updateThresholds(conf: conf, iou: iou, numItems: maxItems)
  }

  override var prefersStatusBarHidden: Bool { true }

  override var prefersHomeIndicatorAutoHidden: Bool { true }
}
