// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app, providing the main user interface for model selection and visualization.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController serves as the primary interface for users to interact with YOLO models.
//  It provides the ability to select different models, tasks (detection, segmentation, classification, etc.),
//  and visualize results in real-time. The controller manages the loading of local and remote models,
//  handles UI updates during model loading and inference, and provides functionality for capturing
//  and sharing detection results. Advanced features include model download progress
//  tracking, and adaptive UI layout for different device orientations.

import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import UIKit
import YOLO

// MARK: - Extensions
extension Result {
  var isSuccess: Bool { if case .success = self { return true } else { return false } }
}

extension Array {
  subscript(safe index: Int) -> Element? {
    return indices.contains(index) ? self[index] : nil
  }
}

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate {

  // MARK: - External Display Support (Optional)
  // NOTE: The following orientation overrides are part of the OPTIONAL external display feature.
  // These features remain dormant until an external display is connected.
  // See ExternalDisplay/ directory for implementation details.

  // Override supported orientations based on external display connection
  override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
    // Use SceneDelegate's state to determine orientation support
    if SceneDelegate.hasExternalDisplay {
      return [.landscapeLeft, .landscapeRight]
    } else {
      return [.portrait, .landscapeLeft, .landscapeRight]
    }
  }

  override var shouldAutorotate: Bool {
    return true
  }

  @IBOutlet weak var yoloView: YOLOView!
  @IBOutlet weak var View0: UIView!
  @IBOutlet weak var segmentedControl: UISegmentedControl!
  @IBOutlet weak var modelSegmentedControl: UISegmentedControl!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var logoImage: UIImageView!

  let selection = UISelectionFeedbackGenerator()

  // Store current loading entry for external display notification (Optional feature)
  var currentLoadingEntry: ModelEntry?

  // Custom model selection button (created programmatically)
  var customModelButton: UIButton!

  // Model version toggle button (YOLO11 ↔ YOLO26)
  var modelVersionToggleButton: UIButton!
  private var modelVersionToggleButtonConstraints: [NSLayoutConstraint] = []

  private let downloadProgressView = UIProgressView(progressViewStyle: .default)
  private let downloadProgressLabel = UILabel()

  private var loadingOverlayView: UIView?

  // MARK: - Constants
  private struct Constants {
    static let defaultTaskIndex = 2  // Detect
    static let tableRowHeight: CGFloat = 30
    static let logoURL = "https://www.ultralytics.com"
    static let progressViewWidth: CGFloat = 200
  }

  // MARK: - Loading State Management
  private func setLoadingState(_ loading: Bool, showOverlay: Bool = false) {
    loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
    view.isUserInteractionEnabled = !loading
    if showOverlay && loading { updateLoadingOverlay(true) }
    if !loading { updateLoadingOverlay(false) }
  }

  private func updateLoadingOverlay(_ show: Bool) {
    if show && loadingOverlayView == nil {
      let overlay = UIView(frame: view.bounds)
      overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)
      view.addSubview(overlay)
      loadingOverlayView = overlay
      view.bringSubviewToFront(downloadProgressView)
      view.bringSubviewToFront(downloadProgressLabel)
    } else if !show {
      loadingOverlayView?.removeFromSuperview()
      loadingOverlayView = nil
    }
  }

  let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
    ("Classify", "ClassifyModels", .classify),
    ("Segment", "SegmentModels", .segment),
    ("Detect", "DetectModels", .detect),
    ("Pose", "PoseModels", .pose),
    ("OBB", "OBBModels", .obb),
  ]

  private var modelsForTask: [String: [String]] = [:]

  var currentModels: [ModelEntry] = []
  private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] =
    [:]

  var currentTask: String = ""
  var currentModelName: String = ""

  // Model version state: true for YOLO26, false for YOLO11
  private var isYOLO26: Bool = true {
    didSet {
      guard isYOLO26 != oldValue else { return }
      // Reload models with new version preference (only if currentTask is set)
      if !currentTask.isEmpty {
        reloadModelEntriesAndLoadFirst(for: currentTask)
      }
      updateModelVersionMenu()
    }
  }

  private var isLoadingModel = false

  override func viewDidLoad() {
    super.viewDidLoad()

    // MARK: External Display Setup (Optional)
    // NOTE: The following external display setup is OPTIONAL and not required for core app functionality.
    // This code enhances the app for external monitor/TV connections and remains dormant when not in use.
    // See ExternalDisplay/ directory and README for more information.

    // Setup external display notifications
    setupExternalDisplayNotifications()

    // Check for already connected external displays
    checkForExternalDisplays()

    // If external display is already connected, ensure YOLOView doesn't interfere
    if hasExternalDisplayConnected() {
      print("External display already connected at startup - deferring camera init")
      yoloView.isHidden = true
    }

    // Sync initial state with YOLOView (after yoloView is initialized)
    // Delay to ensure yoloView is ready
    // Setup segmented control and load models
    segmentedControl.removeAllSegments()
    tasks.enumerated().forEach { index, task in
      segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
      modelsForTask[task.name] = getModelFiles(in: task.folder)
    }

    setupModelSegmentedControl()
    setupCustomModelButton()

    if tasks.indices.contains(Constants.defaultTaskIndex) {
      segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
      currentTask = tasks[Constants.defaultTaskIndex].name

      // Always load models initially - external display handling will stop camera if needed
      reloadModelEntriesAndLoadFirst(for: currentTask)

      // Check for external display after initial setup
      if hasExternalDisplayConnected() {
        print("External display may be connected at startup - will be handled by notifications")
      }
    }

    // Setup gestures and delegates
    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    yoloView.delegate = self
    [yoloView.labelName, yoloView.labelFPS].forEach { $0?.isHidden = true }

    // Add target to sliders to monitor changes
    yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderNumItems.addTarget(
      self, action: #selector(sliderValueChanged), for: .valueChanged)

    // Setup labels and version
    [labelName, labelFPS, labelVersion].forEach {
      $0?.textColor = .white
      $0?.overrideUserInterfaceStyle = .dark
    }
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
      let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    {
      labelVersion.text = "v\(version) (\(build))"
    }

    // Setup model version toggle button
    setupModelVersionToggleButton()

    // Setup progress views
    [downloadProgressView, downloadProgressLabel].forEach {
      $0.isHidden = true
      $0.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview($0)
    }
    downloadProgressLabel.textAlignment = .center
    downloadProgressLabel.textColor = .systemGray
    downloadProgressLabel.font = .systemFont(ofSize: 14)

    NSLayoutConstraint.activate([
      downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      downloadProgressView.topAnchor.constraint(
        equalTo: activityIndicator.bottomAnchor, constant: 8),
      downloadProgressView.widthAnchor.constraint(equalToConstant: Constants.progressViewWidth),
      downloadProgressView.heightAnchor.constraint(equalToConstant: 2),
      downloadProgressLabel.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
      downloadProgressLabel.topAnchor.constraint(
        equalTo: downloadProgressView.bottomAnchor, constant: 8),
    ])

    ModelDownloadManager.shared.progressHandler = { [weak self] progress in
      guard let self = self else { return }
      DispatchQueue.main.async {
        self.downloadProgressView.progress = Float(progress)
        self.downloadProgressLabel.isHidden = false
        let percentage = Int(progress * 100)
        self.downloadProgressLabel.text = "Downloading \(percentage)%"
      }
    }
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    view.overrideUserInterfaceStyle = .dark
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
      let fileURLs = try? FileManager.default.contentsOfDirectory(
        at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
      )
    else { return [] }

    let modelFiles =
      fileURLs
      .filter { ["mlmodel", "mlpackage"].contains($0.pathExtension) }
      .map { $0.lastPathComponent }

    return folderName == "DetectModels" ? reorderDetectionModels(modelFiles) : modelFiles.sorted()
  }

  private func reorderDetectionModels(_ fileNames: [String]) -> [String] {
    let order: [Character: Int] = ["n": 0, "m": 1, "s": 2, "l": 3, "x": 4]
    let (official, custom) = fileNames.reduce(into: ([String](), [String]())) { result, name in
      let base = (name as NSString).deletingPathExtension.lowercased()
      base.hasPrefix("yolo") && order[base.last ?? "z"] != nil
        ? result.0.append(name) : result.1.append(name)
    }
    return custom.sorted()
      + official.sorted {
        order[($0 as NSString).deletingPathExtension.lowercased().last ?? "z"] ?? 99 < order[
          ($1 as NSString).deletingPathExtension.lowercased().last ?? "z"] ?? 99
      }
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    currentModels = makeModelEntries(for: taskName)
    let modelTuples = currentModels.map { ($0.identifier, $0.remoteURL, $0.isLocalBundle) }
    standardModels = ModelSelectionManager.categorizeModels(
      from: modelTuples, preferYOLO26: isYOLO26)

    let yoloTask = tasks.first(where: { $0.name == taskName })?.yoloTask ?? .detect
    ModelSelectionManager.setupSegmentedControl(
      modelSegmentedControl, standardModels: standardModels, currentTask: yoloTask)

    if let firstSize = ModelSelectionManager.ModelSize.allCases.first,
      let model = standardModels[firstSize]
    {
      let entry = ModelEntry(
        displayName: (model.name as NSString).deletingPathExtension,
        identifier: model.name,
        isLocalBundle: model.isLocal,
        isRemote: model.url != nil,
        remoteURL: model.url
      )
      loadModel(entry: entry, forTask: taskName)
    }
  }

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    let localFileNames = modelsForTask[taskName] ?? []
    let localEntries = localFileNames.map { fileName -> ModelEntry in
      let display = (fileName as NSString).deletingPathExtension
      return ModelEntry(
        displayName: display,
        identifier: fileName,
        isLocalBundle: true,
        isRemote: false,
        remoteURL: nil
      )
    }

    // Get local model names for filtering
    let localModelNames = Set(localEntries.map { $0.displayName.lowercased() })

    let remoteList = remoteModelsInfo[taskName] ?? []
    let remoteEntries = remoteList.compactMap { (modelName, url) -> ModelEntry? in
      // Only include remote models if no local model with the same name exists
      guard !localModelNames.contains(modelName.lowercased()) else { return nil }

      return ModelEntry(
        displayName: modelName,
        identifier: modelName,
        isLocalBundle: false,
        isRemote: true,
        remoteURL: url
      )
    }

    return localEntries + remoteEntries
  }

  func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else {
      print("Model is already loading. Please wait.")
      return
    }

    // Cancel any in-progress downloads to prevent conflicts
    ModelDownloadManager.shared.cancelCurrentDownload()

    isLoadingModel = true

    // Check if external display is connected
    let hasExternalDisplay = hasExternalDisplayConnected() || SceneDelegate.hasExternalDisplay

    // Only reset YOLOView if no external display is connected
    if !hasExternalDisplay {
      yoloView.resetLayers()
      yoloView.setInferenceFlag(ok: false)
    }

    setLoadingState(true, showOverlay: true)
    resetDownloadProgress()

    // Store current entry for external display notification
    currentLoadingEntry = entry

    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect

    if entry.isLocalBundle {
      DispatchQueue.global().async { [weak self] in
        guard let self = self else { return }

        guard let folderURL = self.tasks.first(where: { $0.name == task })?.folder,
          let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
        else {
          Task { @MainActor [weak self] in
            self?.finishLoadingModel(success: false, modelName: entry.displayName)
          }
          return
        }

        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        Task { @MainActor [weak self] in
          guard let self = self else { return }
          self.downloadProgressLabel.isHidden = false
          self.downloadProgressLabel.text = "Loading \(entry.displayName)"

          // Check if external display is connected
          let hasExternalDisplay = hasExternalDisplayConnected() || SceneDelegate.hasExternalDisplay

          if hasExternalDisplay {
            // External display is connected - skip YOLOView loading, just notify external display
            print("External display connected - skipping main YOLOView model load")
            self.finishLoadingModel(success: true, modelName: entry.displayName)
          } else {
            // Normal model loading on main YOLOView
            self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
              Task { @MainActor [weak self] in
                guard let self = self else { return }
                switch result {
                case .success():
                  self.finishLoadingModel(success: true, modelName: entry.displayName)
                case .failure(let error):
                  print(error)
                  self.finishLoadingModel(success: false, modelName: entry.displayName)
                }
              }
            }
          }
        }
      }
    } else {
      let key = entry.identifier  // "yolov8n", "yolov8m-seg", etc.

      if ModelCacheManager.shared.isModelDownloaded(key: key) {
        loadCachedModelAndSetToYOLOView(
          key: key, yoloTask: yoloTask, displayName: entry.displayName)
      } else {
        guard let remoteURL = entry.remoteURL else {
          Task { @MainActor [weak self] in
            self?.finishLoadingModel(success: false, modelName: entry.displayName)
          }
          return
        }

        self.downloadProgressView.progress = 0.0
        self.downloadProgressView.isHidden = false
        self.downloadProgressLabel.isHidden = false

        // Set initial downloading message with proper model name
        self.downloadProgressLabel.text = "Downloading \(processString(entry.displayName))"

        let localZipFileName = remoteURL.lastPathComponent  // ex. "yolov8n.mlpackage.zip"

        ModelCacheManager.shared.loadModel(
          from: localZipFileName,
          remoteURL: remoteURL,
          key: key
        ) { [weak self] mlModel, loadedKey in
          guard let self = self else { return }
          if mlModel == nil {
            Task { @MainActor [weak self] in
              self?.finishLoadingModel(success: false, modelName: entry.displayName)
            }
            return
          }
          self.loadCachedModelAndSetToYOLOView(
            key: loadedKey,
            yoloTask: yoloTask,
            displayName: entry.displayName)
        }
      }
    }
  }

  private func loadCachedModelAndSetToYOLOView(key: String, yoloTask: YOLOTask, displayName: String)
  {
    let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[
      0]
    let localModelURL = documentsDirectory.appendingPathComponent(key).appendingPathExtension(
      "mlmodelc")

    Task { @MainActor [weak self] in
      guard let self = self else { return }
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"

      // Check if external display is connected
      let hasExternalDisplay = hasExternalDisplayConnected() || SceneDelegate.hasExternalDisplay

      if hasExternalDisplay {
        // External display is connected - skip YOLOView loading, just notify external display
        print("External display connected - skipping main YOLOView cached model load")
        self.finishLoadingModel(success: true, modelName: displayName)
      } else {
        // Normal model loading on main YOLOView
        self.yoloView.setModel(modelPathOrName: localModelURL.path, task: yoloTask) { result in
          Task { @MainActor [weak self] in
            guard let self = self else { return }
            switch result {
            case .success():
              self.finishLoadingModel(success: true, modelName: displayName)
            case .failure(let error):
              print(error)
              self.finishLoadingModel(success: false, modelName: displayName)
            }
          }
        }
      }
    }
  }

  private func resetDownloadProgress() {
    downloadProgressView.progress = 0.0
    downloadProgressLabel.text = ""
    [downloadProgressView, downloadProgressLabel].forEach { $0.isHidden = true }
  }

  @MainActor
  private func finishLoadingModel(success: Bool, modelName: String) {
    setLoadingState(false)
    isLoadingModel = false
    resetDownloadProgress()

    if success {
      let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect

      ModelSelectionManager.setupSegmentedControl(
        modelSegmentedControl,
        standardModels: standardModels,
        currentTask: yoloTask,
        preserveSelection: true
      )

      ModelSelectionManager.updateSegmentAppearance(
        modelSegmentedControl,
        standardModels: standardModels,
        currentTask: yoloTask
      )
    }

    // Notify external display of model change (Optional feature)
    if success {
      // Update currentModelName
      currentModelName = processString(modelName)

      let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect

      // Determine the correct model path for external display
      var fullModelPath = ""

      // Use the stored entry from loadModel
      if let entry = currentLoadingEntry {
        if entry.isLocalBundle {
          // For local bundle models
          if let folderURL = tasks.first(where: { $0.name == currentTask })?.folder,
            let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
          {
            let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
            fullModelPath = modelURL.path
          }
        } else {
          // For remote/downloaded models, we need to pass the identifier only
          // The external display will handle loading from cache
          fullModelPath = entry.identifier

          // Verify the cached model exists locally first
          let documentsDirectory = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask)[0]
          let localModelURL =
            documentsDirectory
            .appendingPathComponent(entry.identifier)
            .appendingPathExtension("mlmodelc")

          if !FileManager.default.fileExists(atPath: localModelURL.path) {
            print("Cached model not found at: \(localModelURL.path)")
            return
          }
        }
      }

      // Only notify if we have a valid path
      if !fullModelPath.isEmpty {
        ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)

        // Also check if external display is waiting for initial model
        checkAndNotifyExternalDisplayIfReady()
      } else {
        print("Could not determine model path for external display")
      }
    }

    // Check if external display is connected
    let hasExternalDisplay = hasExternalDisplayConnected() || SceneDelegate.hasExternalDisplay

    // Only set inference flag on YOLOView if no external display
    if !hasExternalDisplay {
      yoloView.setInferenceFlag(ok: success)
    }

    if success {
      // currentModelName is already set above in the notification section
      labelName.text = processString(modelName)
    }
  }

  @IBAction func vibrate(_ sender: Any) { selection.selectionChanged() }

  @IBAction func indexChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()
    guard tasks.indices.contains(sender.selectedSegmentIndex) else { return }

    let newTask = tasks[sender.selectedSegmentIndex].name

    if (modelsForTask[newTask]?.isEmpty ?? true) && (remoteModelsInfo[newTask]?.isEmpty ?? true) {
      let alert = UIAlertController(
        title: "\(newTask) Models not found",
        message: "Please add or define models for \(newTask).", preferredStyle: .alert)
      alert.addAction(
        UIAlertAction(title: "OK", style: .cancel) { _ in alert.dismiss(animated: true) })
      present(alert, animated: true)
      sender.selectedSegmentIndex = tasks.firstIndex { $0.name == currentTask } ?? 0
      return
    }

    currentTask = newTask

    // Notify external display of task change immediately (Optional external display feature)
    NotificationCenter.default.post(
      name: .taskDidChange,
      object: nil,
      userInfo: ["task": newTask]
    )
    reloadModelEntriesAndLoadFirst(for: currentTask)
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: Constants.logoURL) {
      UIApplication.shared.open(link)
    }
  }

  /// Setup model version selection button (shows menu to select YOLO11 or YOLO26)
  private func setupModelVersionToggleButton() {
    modelVersionToggleButton = UIButton(type: .system)
    let config = UIImage.SymbolConfiguration(pointSize: 14, weight: .medium, scale: .default)
    modelVersionToggleButton.setImage(
      UIImage(systemName: "chevron.down", withConfiguration: config), for: .normal)
    modelVersionToggleButton.tintColor = .white
    // Match the transparent/grayish style of the model segmented control
    modelVersionToggleButton.backgroundColor = .systemBackground.withAlphaComponent(0.1)
    modelVersionToggleButton.layer.cornerRadius = 12
    modelVersionToggleButton.layer.borderWidth = 1
    modelVersionToggleButton.layer.borderColor = UIColor.systemGray.cgColor
    modelVersionToggleButton.translatesAutoresizingMaskIntoConstraints = false
    modelVersionToggleButton.isHidden = false
    modelVersionToggleButton.alpha = 1.0
    modelVersionToggleButton.showsMenuAsPrimaryAction = true
    view.addSubview(modelVersionToggleButton)
    view.bringSubviewToFront(modelVersionToggleButton)
    updateModelVersionMenu()

    // Position button next to labelName - use viewDidLayoutSubviews to set constraints after layout
    DispatchQueue.main.async { [weak self] in
      self?.updateModelVersionToggleButtonPosition()
    }
  }

  /// Update toggle button position relative to labelName (next to it, not at trailing edge)
  private func updateModelVersionToggleButtonPosition() {
    guard let labelName = labelName, let button = modelVersionToggleButton else { return }

    // Only set constraints if they haven't been set yet
    if modelVersionToggleButtonConstraints.isEmpty {
      modelVersionToggleButtonConstraints = [
        button.leadingAnchor.constraint(equalTo: labelName.trailingAnchor, constant: 8),
        button.centerYAnchor.constraint(equalTo: labelName.centerYAnchor),
        button.widthAnchor.constraint(equalToConstant: 24),
        button.heightAnchor.constraint(equalToConstant: 24),
      ]
      NSLayoutConstraint.activate(modelVersionToggleButtonConstraints)
    }
  }

  /// Build and assign the model version selection menu (YOLO11 or YOLO26)
  private func updateModelVersionMenu() {
    guard modelVersionToggleButton != nil else { return }

    let yolo26Action = UIAction(
      title: "YOLO26",
      state: isYOLO26 ? .on : .off
    ) { [weak self] _ in
      self?.selection.selectionChanged()
      self?.isYOLO26 = true
    }

    let yolo11Action = UIAction(
      title: "YOLO11",
      state: isYOLO26 ? .off : .on
    ) { [weak self] _ in
      self?.selection.selectionChanged()
      self?.isYOLO26 = false
    }

    modelVersionToggleButton.menu = UIMenu(
      title: "Select Model Version",
      options: [.singleSelection],
      children: [yolo26Action, yolo11Action]
    )
  }

  private func setupModelSegmentedControl() {
    modelSegmentedControl.isHidden = false
    modelSegmentedControl.overrideUserInterfaceStyle = .dark
    modelSegmentedControl.apportionsSegmentWidthsByContent = true
    modelSegmentedControl.addTarget(
      self, action: #selector(modelSizeChanged(_:)), for: .valueChanged)

    modelSegmentedControl.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      modelSegmentedControl.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
      modelSegmentedControl.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
    ])
  }

  private func setupCustomModelButton() {
    customModelButton = UIButton(type: .system)
    customModelButton.setTitle("Custom", for: .normal)
    customModelButton.titleLabel?.font = UIFont.systemFont(ofSize: 13)
    customModelButton.setTitleColor(.white, for: .normal)
    customModelButton.setTitleColor(.systemBlue, for: .selected)
    customModelButton.backgroundColor = .systemBackground.withAlphaComponent(0.1)
    customModelButton.layer.cornerRadius = 8
    customModelButton.layer.borderWidth = 1
    customModelButton.layer.borderColor = UIColor.systemGray.cgColor
    customModelButton.addTarget(
      self, action: #selector(customModelButtonTapped), for: .touchUpInside)
    customModelButton.translatesAutoresizingMaskIntoConstraints = false

    View0.addSubview(customModelButton)

    modelSegmentedControl.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      modelSegmentedControl.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
      modelSegmentedControl.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
    ])
  }

  // MARK: - Actions
  @objc func customModelButtonTapped() {
    selection.selectionChanged()
    // Placeholder action for custom model selection; integrate picker if needed
  }

  func updateModelSegmentedControlAppearance() {
    guard modelSegmentedControl != nil else { return }

    modelSegmentedControl.overrideUserInterfaceStyle = .dark
    modelSegmentedControl.backgroundColor = .clear

    let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect
    ModelSelectionManager.updateSegmentAppearance(
      modelSegmentedControl, standardModels: standardModels, currentTask: yoloTask)
  }

  @objc private func modelSizeChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()

    if sender.selectedSegmentIndex < ModelSelectionManager.ModelSize.allCases.count {
      let size = ModelSelectionManager.ModelSize.allCases[sender.selectedSegmentIndex]
      if let model = standardModels[size] {
        let entry = ModelEntry(
          displayName: (model.name as NSString).deletingPathExtension,
          identifier: model.name,
          isLocalBundle: model.isLocal,
          isRemote: model.url != nil,
          remoteURL: model.url
        )
        loadModel(entry: entry, forTask: currentTask)
      }
    }
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    adjustLayoutForExternalDisplayIfNeeded()
    // Set button position constraints after layout (only once)
    updateModelVersionToggleButtonPosition()
  }

  @objc func shareButtonTapped() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] image in
      guard let self = self, let image = image else { return print("error capturing photo") }
      DispatchQueue.main.async { [weak self] in
        guard let self = self else { return }
        let vc = UIActivityViewController(activityItems: [image], applicationActivities: nil)
        vc.popoverPresentationController?.sourceView = self.View0
        self.present(vc, animated: true)
      }
    }
  }

  @objc func sliderValueChanged(_ sender: UISlider) {
    // Send threshold values to external display (Optional external display feature)
    let conf = Double(round(100 * yoloView.sliderConf.value)) / 100
    let iou = Double(round(100 * yoloView.sliderIoU.value)) / 100
    let maxItems = Int(yoloView.sliderNumItems.value)

    NotificationCenter.default.post(
      name: .thresholdDidChange,
      object: nil,
      userInfo: [
        "conf": conf,
        "iou": iou,
        "maxItems": maxItems,
      ]
    )

  }

  deinit {
    NotificationCenter.default.removeObserver(self)
    ModelDownloadManager.shared.progressHandler = nil
  }

}

// MARK: - YOLOViewDelegate
extension ViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { [weak self] in
      self?.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
      self?.labelFPS.textColor = .white
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    DispatchQueue.main.async { [weak self] in
      guard self != nil else { return }
      // Share results with external display (Optional external display feature)
      ExternalDisplayManager.shared.shareResults(result)

      // Also send via notification for direct communication (Optional external display feature)
      NotificationCenter.default.post(
        name: .yoloResultsAvailable,
        object: nil,
        userInfo: ["result": result]
      )
    }
  }

}
