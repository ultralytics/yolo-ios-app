// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import CoreMedia
import UIKit
import YOLO

class ExternalViewController: UIViewController, YOLOViewDelegate {

  private var yoloView: YOLOView?
  private var isInitialized = false
  private var currentTask: YOLOTask = .detect
  private var currentModelName: String = "yolo26n"

  // UI Elements with proper scaling
  private var labelName: UILabel!
  private var labelFPS: UILabel!
  private var segmentedControl: UISegmentedControl!
  private var logoImageView: UIImageView!

  // Task info
  private let tasks: [(name: String, value: YOLOTask)] = [
    ("Classify", .classify),
    ("Segment", .segment),
    ("Detect", .detect),
    ("Pose", .pose),
    ("OBB", .obb),
  ]

  override func viewDidLoad() {
    super.viewDidLoad()

    view.backgroundColor = .black
    edgesForExtendedLayout = .all
    extendedLayoutIncludesOpaqueBars = true

    setupControlUI()
    setupNotifications()
  }

  private func setupControlUI() {
    let screenSize = view.bounds.size
    let scaleFactor = calculateScaleFactor(for: screenSize)

    // Proportional font sizes
    let baseFontSizeModelName = screenSize.height * 0.1
    let baseFontSizeFPS = screenSize.height * 0.04

    // Model name label
    labelName = createLabel(
      text: currentModelName,
      fontSize: baseFontSizeModelName,
      weight: .bold
    )
    view.addSubview(labelName)

    // FPS label
    labelFPS = createLabel(
      text: "0.0 FPS - 0.0 ms",
      fontSize: baseFontSizeFPS,
      weight: .medium
    )
    view.addSubview(labelFPS)

    // Task segmented control
    segmentedControl = createSegmentedControl()
    view.addSubview(segmentedControl)

    // Logo ImageView
    logoImageView = UIImageView(image: UIImage(named: "ultralytics_yolo_logotype"))
    logoImageView.contentMode = .scaleAspectFit
    logoImageView.alpha = 1.0
    logoImageView.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(logoImageView)

    setupConstraints(scaleFactor: scaleFactor)
  }

  private func createLabel(text: String, fontSize: CGFloat, weight: UIFont.Weight) -> UILabel {
    let label = UILabel()
    label.text = text
    label.textAlignment = .center
    label.font = .systemFont(ofSize: fontSize, weight: weight)
    label.textColor = .white
    label.adjustsFontSizeToFitWidth = true
    label.minimumScaleFactor = 0.5
    label.numberOfLines = 1
    label.translatesAutoresizingMaskIntoConstraints = false
    label.setContentHuggingPriority(.required, for: .vertical)
    label.setContentCompressionResistancePriority(.required, for: .vertical)
    return label
  }

  private func createSegmentedControl() -> UISegmentedControl {
    let control = UISegmentedControl()
    for (index, taskInfo) in tasks.enumerated() {
      control.insertSegment(withTitle: taskInfo.name, at: index, animated: false)
    }
    control.selectedSegmentIndex = 2  // Default to Detect

    // Styling
    control.backgroundColor = UIColor(white: 0.2, alpha: 0.3)
    control.selectedSegmentTintColor = UIColor(white: 0.4, alpha: 0.8)
    control.layer.cornerRadius = 12
    control.layer.masksToBounds = true

    let fontSize: CGFloat = 36
    control.setTitleTextAttributes(
      [
        .foregroundColor: UIColor.white,
        .font: UIFont.systemFont(ofSize: fontSize, weight: .semibold),
      ], for: .selected)
    control.setTitleTextAttributes(
      [
        .foregroundColor: UIColor.lightGray,
        .font: UIFont.systemFont(ofSize: fontSize, weight: .medium),
      ], for: .normal)

    control.isUserInteractionEnabled = false
    control.translatesAutoresizingMaskIntoConstraints = false
    control.setContentHuggingPriority(.required, for: .vertical)
    control.setContentCompressionResistancePriority(.required, for: .vertical)
    control.isHidden = true
    return control
  }

  private func calculateScaleFactor(for screenSize: CGSize) -> CGFloat {
    let baseSize: CGFloat = 375.0
    let rawScale = max(screenSize.width, screenSize.height) / baseSize
    let scaleFactor = 1.0 + (rawScale - 1.0) * 1.2
    return max(2.5, min(scaleFactor, 10.0))
  }

  private func setupConstraints(scaleFactor: CGFloat) {
    let margin: CGFloat = 20

    NSLayoutConstraint.activate([
      labelName.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      labelName.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: margin),
      labelName.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.9),
      labelName.heightAnchor.constraint(equalTo: view.heightAnchor, multiplier: 0.1),

      labelFPS.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      labelFPS.topAnchor.constraint(equalTo: labelName.bottomAnchor, constant: 20),
      labelFPS.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.9),
      labelFPS.heightAnchor.constraint(equalTo: view.heightAnchor, multiplier: 0.04),

      segmentedControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      segmentedControl.topAnchor.constraint(equalTo: labelFPS.bottomAnchor, constant: 30),
      segmentedControl.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.7),
      segmentedControl.heightAnchor.constraint(equalToConstant: 60),

      logoImageView.trailingAnchor.constraint(
        equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -margin),
      logoImageView.bottomAnchor.constraint(
        equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -margin),
      logoImageView.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.2),
      logoImageView.heightAnchor.constraint(equalTo: logoImageView.widthAnchor, multiplier: 0.3),
    ])
  }

  private func hideYOLOViewControls() {
    guard let yoloView = yoloView else { return }

    let controlsToRemove = [
      yoloView.sliderNumItems, yoloView.labelSliderNumItems,
      yoloView.sliderConf, yoloView.labelSliderConf,
      yoloView.sliderIoU, yoloView.labelSliderIoU,
      yoloView.labelName, yoloView.labelFPS,
      yoloView.toolbar,
    ]

    controlsToRemove.forEach {
      $0.isHidden = true
      $0.removeFromSuperview()
    }

    [
      yoloView.labelZoom, yoloView.activityIndicator,
      yoloView.playButton, yoloView.pauseButton,
      yoloView.switchCameraButton,
    ].forEach { $0.isHidden = true }
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)

    guard !isInitialized else { return }
    isInitialized = true

    setupYOLOView()
  }

  private func setupYOLOView() {
    // Create YOLOView without model initially - will be set when main app notifies
    yoloView = YOLOView(frame: view.bounds)
    yoloView?.delegate = self
    yoloView?.backgroundColor = .clear

    guard let yoloView = yoloView else { return }

    view.insertSubview(yoloView, at: 0)

    yoloView.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      yoloView.topAnchor.constraint(equalTo: view.topAnchor),
      yoloView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      yoloView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      yoloView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
    ])

    // Re-add UI elements on top
    [logoImageView, labelName, labelFPS].forEach {
      $0?.removeFromSuperview()
      if let view = $0 { self.view.addSubview(view) }
    }

    let scaleFactor = calculateScaleFactor(for: view.bounds.size)
    setupConstraints(scaleFactor: scaleFactor)

    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
      self.hideYOLOViewControls()
      // Don't load initial model - wait for main app to notify us
      NotificationCenter.default.post(name: .externalDisplayReady, object: nil)
    }
  }

  private func setupNotifications() {
    // Listen for model changes
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleModelChange(_:)),
      name: .modelDidChange,
      object: nil
    )

    // Listen for threshold changes from iPhone
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleThresholdChange(_:)),
      name: .thresholdDidChange,
      object: nil
    )

    // Listen for task changes from iPhone
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleTaskChange(_:)),
      name: .taskDidChange,
      object: nil
    )
  }

  @objc private func handleModelChange(_ notification: Notification) {
    guard let userInfo = notification.userInfo,
      let taskString = userInfo["task"] as? String,
      let modelName = userInfo["modelName"] as? String
    else { return }

    let taskMap: [String: YOLOTask] = [
      "detect": .detect, "segment": .segment,
      "classify": .classify, "pose": .pose, "obb": .obb,
    ]
    let task = taskMap[taskString] ?? .detect

    currentTask = task
    currentModelName = modelName

    if let taskIndex = tasks.firstIndex(where: { $0.value == task }) {
      DispatchQueue.main.async { [weak self] in
        self?.segmentedControl.selectedSegmentIndex = taskIndex
      }
    }

    // Determine the actual model path to use
    var actualModelPath = modelName

    // Check if this is a full path (local bundle model) or just an identifier (downloaded model)
    if !modelName.hasPrefix("/") && !modelName.contains(".mlpackage")
      && !modelName.contains(".mlmodel")
    {
      // This is a downloaded model identifier
      // First check if the model is actually downloaded
      if ModelCacheManager.shared.isModelDownloaded(key: modelName) {
        // Construct the full path for the downloaded model
        let documentsDirectory = FileManager.default.urls(
          for: .documentDirectory, in: .userDomainMask)[0]
        let localModelURL = documentsDirectory.appendingPathComponent(modelName)
          .appendingPathExtension("mlmodelc")
        actualModelPath = localModelURL.path
        print("📱 External display loading cached model from: \(actualModelPath)")
      } else {
        print("❌ Model not downloaded yet: \(modelName)")
        return  // Exit early if model is not downloaded
      }
    } else {
      print("📱 External display loading bundle model from: \(actualModelPath)")
    }

    let capturedModelPath = actualModelPath
    yoloView?.setModel(modelPathOrName: capturedModelPath, task: task) { [weak self] result in
      guard case .success = result else {
        print("❌ Failed to load model on external display: \(capturedModelPath)")
        return
      }

      DispatchQueue.main.async {
        self?.updateModelNameLabel()
        self?.yoloView?.setNeedsDisplay()
        self?.yoloView?.layoutIfNeeded()
      }
    }
  }

  private func updateModelNameLabel() {
    let modelDisplayName = (currentModelName as NSString).lastPathComponent
    var nameWithoutExtension = (modelDisplayName as NSString).deletingPathExtension

    if nameWithoutExtension.hasSuffix(".mlmodelc") {
      nameWithoutExtension = (nameWithoutExtension as NSString).deletingPathExtension
    }

    // Use processString to properly format the model name (handles YOLO11n-seg -> YOLO11n Seg, etc.)
    labelName.text = processString(nameWithoutExtension)
  }

  @objc private func handleThresholdChange(_ notification: Notification) {
    guard let userInfo = notification.userInfo,
      let conf = userInfo["conf"] as? Double,
      let iou = userInfo["iou"] as? Double,
      let maxItems = userInfo["maxItems"] as? Int,
      let yoloView = yoloView
    else {
      return
    }

    // Use the new updateThresholds method to properly sync all threshold values
    yoloView.updateThresholds(conf: conf, iou: iou, numItems: maxItems)
  }

  @objc private func handleTaskChange(_ notification: Notification) {
    guard let userInfo = notification.userInfo,
      let taskName = userInfo["task"] as? String
    else {
      return
    }

    // Find the task in our tasks array and update segment control
    if let taskIndex = tasks.firstIndex(where: { $0.name == taskName }) {
      DispatchQueue.main.async {
        self.segmentedControl.selectedSegmentIndex = taskIndex
      }
    }
  }

  override var prefersStatusBarHidden: Bool {
    return true
  }

  override var prefersHomeIndicatorAutoHidden: Bool {
    return true
  }

  // Support all orientations for external display
  override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
    return .all
  }

  override var shouldAutorotate: Bool {
    return true
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }
}

// MARK: - YOLOViewDelegate
extension ExternalViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { [weak self] in
      self?.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    let detectionCount: Int

    switch currentTask {
    case .pose:
      detectionCount = result.keypointsList.count
    case .obb:
      detectionCount = result.obb.count
    case .classify:
      detectionCount = result.probs != nil ? 1 : 0
    default:
      detectionCount = result.boxes.count
    }

    NotificationCenter.default.post(
      name: .detectionCountDidUpdate,
      object: nil,
      userInfo: ["count": detectionCount]
    )
  }

}
