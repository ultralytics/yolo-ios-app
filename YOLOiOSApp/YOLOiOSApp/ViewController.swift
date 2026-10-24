// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app and provides the main user interface for model selection and
//  visualization.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController is the primary interface for users to interact with YOLO models. Users can select different
//  models and tasks (detection, segmentation, semantic segmentation, depth, classification, pose, OBB) and visualize
//  results in real-time. The controller manages loading of local and remote models, updates UI state during loading and
//  inference, and handles capturing and sharing detection results. It also tracks model download progress and adapts
//  the layout to different device orientations.

import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import UIKit
import UltralyticsYOLO

let appTasks: [(name: String, shortName: String, folder: String, yoloTask: YOLOTask)] = [
  ("Detect", "Det", "Models/Detect", .detect),
  ("Segment", "Seg", "Models/Segment", .segment),
  ("Semantic", "Sem", "Models/Semantic", .semantic),
  ("Depth", "Depth", "Models/Depth", .depth),
  ("Classify", "Cls", "Models/Classify", .classify),
  ("Pose", "Pose", "Models/Pose", .pose),
  ("OBB", "OBB", "Models/OBB", .obb),
]

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate {

  // MARK: - External Display Support (Optional)
  // External display features remain dormant until a display is connected. See ExternalDisplay/ for details.

  @IBOutlet weak var yoloView: YOLOView!
  @IBOutlet weak var View0: UIView!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  /// Latest "x FPS - y ms" line and pre/inference/post breakdown, rendered together in `labelFPS`.
  private var fpsText = ""
  private var breakdownText = ""
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var logoImage: UIImageView!

  let selection = UISelectionFeedbackGenerator()

  // Tracks the currently loading model for external display notification.
  var currentLoadingEntry: ModelEntry?

  private let downloadProgressView = UIProgressView(progressViewStyle: .default)
  private let downloadProgressLabel = UILabel()
  private let cancelLoadingButton = UIButton(type: .system)
  private lazy var modelsButton = ModelHeaderControl(title: labelName)
  private let scanButton = UIButton(type: .system)
  private let qrOverlay = QRScanOverlayView()
  private var qrCamera: YOLOView? { didSet { updateActivityState() } }

  private var loadingOverlayView: UIView?

  // MARK: - Constants
  private struct Constants {
    static let defaultTask = YOLOTask.detect
    static let logoURL = "https://www.ultralytics.com"
    static let progressViewWidth: CGFloat = 200
  }

  // MARK: - Loading State Management
  private func setLoadingState(_ loading: Bool, showOverlay: Bool = false, canCancel: Bool = false)
  {
    if showOverlay && loading { updateLoadingOverlay(true) }
    if !loading { updateLoadingOverlay(false) }
    cancelLoadingButton.isHidden = !(loading && canCancel)
  }

  private func updateActivityState() {
    let scanning = qrCamera != nil
    scanButton.isEnabled = !isLoadingModel
    scanButton.configuration?.baseForegroundColor =
      scanning ? .systemYellow : yoloView.shareButton.tintColor
    scanButton.configuration?.background.backgroundColor =
      scanning ? UIColor.systemYellow.withAlphaComponent(0.16) : .clear
    scanButton.accessibilityTraits = scanning ? [.button, .selected] : .button
    modelsButton.isEnabled = !isLoadingModel && !scanning
    qrOverlay.isHidden = !scanning
    [
      labelFPS, logoImage, yoloView.labelSliderConf, yoloView.sliderConf, yoloView.labelSliderIoU,
      yoloView.sliderIoU,
    ]
    .forEach { $0?.isHidden = scanning }
    if isLoadingModel {
      activityIndicator.startAnimating()
    } else {
      activityIndicator.stopAnimating()
    }
  }

  private func updateLoadingOverlay(_ show: Bool) {
    if show && loadingOverlayView == nil {
      let overlay = UIView(frame: view.bounds)
      overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)
      overlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
      view.addSubview(overlay)
      loadingOverlayView = overlay
      view.bringSubviewToFront(activityIndicator)
      view.bringSubviewToFront(downloadProgressView)
      view.bringSubviewToFront(downloadProgressLabel)
      view.bringSubviewToFront(cancelLoadingButton)
    } else if !show {
      loadingOverlayView?.removeFromSuperview()
      loadingOverlayView = nil
    }
  }

  private var modelsForTask: [String: [String]] = [:]

  var currentModels: [ModelEntry] = []

  var currentTask: String = ""
  var currentModelName: String = ""
  var currentModelIdentifier: String = ""
  var initialImportURL: URL?

  private var isLoadingModel = false { didSet { updateActivityState() } }
  private var currentLoadingTask: String?

  override func viewDidLoad() {
    super.viewDidLoad()

    // Optional external display setup — dormant unless a monitor/TV is connected. See ExternalDisplay/ for details.
    setupExternalDisplayNotifications()

    appTasks.forEach { modelsForTask[$0.name] = getModelFiles(in: $0.folder) }

    setupModelActions()

    currentTask = appTasks.first { $0.yoloTask == Constants.defaultTask }!.name

    // External display handling will stop the local camera if needed.
    if initialImportURL != nil {
      showModelEntries(for: currentTask)
    } else {
      loadSavedModelOrFirstOfficial()
    }

    // Returning from Settings with the Core AI setting changed: relist models in the new format and load the current
    // task's first model in it. A change made while a model loads is picked up on the next return to the foreground.
    NotificationCenter.default.addObserver(
      forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main
    ) { [weak self] _ in
      guard let self = self, !self.isLoadingModel,
        preferredModelExtension != remoteModelExtension
      else { return }
      remoteModelExtension = preferredModelExtension
      appTasks.forEach { self.modelsForTask[$0.name] = self.getModelFiles(in: $0.folder) }
      if self.currentModels.first(where: { $0.identifier == self.currentModelIdentifier })?
        .installedURL != nil
      {
        self.showModelEntries(for: self.currentTask)
      } else {
        self.reloadModelEntriesAndLoadFirst(for: self.currentTask)
      }
    }

    // Wire up gestures and delegates.
    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    yoloView.delegate = self
    [yoloView.labelName, yoloView.labelFPS, yoloView.labelBreakdown].forEach { $0?.isHidden = true }

    // Observe slider changes to forward thresholds to an external display.
    yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)

    // Style labels and stamp the app version.
    [labelName, labelFPS, labelVersion].forEach {
      $0?.textColor = .white
      $0?.overrideUserInterfaceStyle = .dark
    }
    labelFPS.numberOfLines = 2
    // Clear the storyboard placeholder text and reserve the final two-line height immediately,
    // so the first inference result doesn't shift the layout below.
    labelName.text = ""
    fpsText = " "
    breakdownText = " "
    renderPerformanceLabel()
    activityIndicator.style = .large
    activityIndicator.color = .white
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
      let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    {
      labelVersion.text = "v\(version) (\(build))"
    }

    // Install the download progress views.
    [downloadProgressView, downloadProgressLabel, cancelLoadingButton].forEach {
      $0.isHidden = true
      $0.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview($0)
    }
    downloadProgressLabel.textAlignment = .center
    downloadProgressLabel.textColor = .white
    downloadProgressLabel.font = .systemFont(ofSize: 15, weight: .medium)
    downloadProgressView.progressTintColor = .white
    downloadProgressView.trackTintColor = UIColor.white.withAlphaComponent(0.25)
    var cancelButtonConfiguration = UIButton.Configuration.tinted()
    cancelButtonConfiguration.title = "Cancel"
    cancelButtonConfiguration.baseForegroundColor = .white
    cancelButtonConfiguration.baseBackgroundColor = .white
    cancelButtonConfiguration.contentInsets = NSDirectionalEdgeInsets(
      top: 8, leading: 16, bottom: 8, trailing: 16)
    cancelLoadingButton.configuration = cancelButtonConfiguration
    cancelLoadingButton.addTarget(self, action: #selector(cancelLoadingModel), for: .touchUpInside)

    NSLayoutConstraint.activate([
      downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      downloadProgressView.topAnchor.constraint(
        equalTo: activityIndicator.bottomAnchor, constant: 8),
      downloadProgressView.widthAnchor.constraint(equalToConstant: Constants.progressViewWidth),
      downloadProgressView.heightAnchor.constraint(equalToConstant: 3),
      downloadProgressLabel.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
      downloadProgressLabel.topAnchor.constraint(
        equalTo: downloadProgressView.bottomAnchor, constant: 8),
      cancelLoadingButton.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
      cancelLoadingButton.topAnchor.constraint(
        equalTo: downloadProgressLabel.bottomAnchor, constant: 10),
    ])

    ModelDownloadManager.shared.progressHandler = { [weak self] progress in
      guard let self = self else { return }
      DispatchQueue.main.async {
        self.downloadProgressView.progress = Float(progress)
        self.downloadProgressLabel.isHidden = false
        let percentage = Int(progress * 100)
        self.downloadProgressLabel.text =
          progress >= 1 ? "Installing model" : "Downloading \(percentage)%"
        self.cancelLoadingButton.isHidden = progress >= 1
      }
    }
    if let url = initialImportURL { importModel(from: url) }
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    view.overrideUserInterfaceStyle = .dark
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
      let fileURLs = try? FileManager.default.contentsOfDirectory(
        at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
      )
    else { return [] }

    let modelFiles =
      fileURLs
      .filter {
        // With the Core AI setting on, the app lists Core AI models only: bundled `.aimodel` and remote `.aimodel.zip`.
        (remoteModelExtension == "aimodel" ? ["aimodel"] : ["mlmodel", "mlpackage"])
          .contains($0.pathExtension)
      }
      .map { $0.lastPathComponent }

    return modelFiles.sorted()
  }

  private func loadSavedModelOrFirstOfficial() {
    if let savedModel = ModelCacheManager.shared.customModels.first(where: {
      $0.identifier == UserDefaults.standard.string(forKey: "selectedCustomModel")
    }), let task = appTasks.first(where: { String(describing: $0.yoloTask) == savedModel.task }) {
      showModelEntries(for: task.name)
      loadModel(entry: savedModel, forTask: task.name)
    } else {
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    showModelEntries(for: taskName)
    if let entry = currentModels.first(where: { $0.installedURL == nil }) {
      loadModel(entry: entry, forTask: taskName)
    }
  }

  private func showModelEntries(for taskName: String) {
    currentModels =
      makeModelEntries(for: taskName)
      + ModelCacheManager.shared.customModels.filter { entry in
        appTasks.first(where: { String(describing: $0.yoloTask) == entry.task })?.name == taskName
      }
  }

  private func restoreCurrentModelSelection() {
    guard !currentTask.isEmpty else { return }
    showModelEntries(for: currentTask)
    if !currentModelName.isEmpty { labelName.text = currentModelName }
  }

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    let task = String(describing: appTasks.first { $0.name == taskName }!.yoloTask)
    let localFileNames = modelsForTask[taskName] ?? []
    let localEntries = localFileNames.map { fileName -> ModelEntry in
      ModelEntry(
        displayName: (fileName as NSString).deletingPathExtension,
        identifier: fileName,
        isLocalBundle: true,
        remoteURL: nil, task: task
      )
    }

    let remoteList = remoteModelsInfo[taskName] ?? []
    let officialEntries = remoteList.map { (modelName, url) -> ModelEntry in
      localEntries.first(where: { $0.displayName.lowercased() == modelName.lowercased() })
        ?? ModelEntry(
          displayName: modelName, identifier: modelName, isLocalBundle: false, remoteURL: url,
          task: task)
    }
    let officialNames = Set(officialEntries.map { $0.displayName.lowercased() })
    return officialEntries
      + localEntries.filter { !officialNames.contains($0.displayName.lowercased()) }
  }

  func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else {
      print("Model is already loading. Please wait.")
      return
    }
    qrCamera?.cancelQRCodeScan()
    isLoadingModel = true

    let needsDownload =
      entry.installedURL == nil && !entry.isLocalBundle
      && !ModelCacheManager.shared.isModelDownloaded(key: entry.identifier)

    // Skip the local YOLOView reset when an external display owns the camera.
    let hasExternalDisplay = hasExternalScreen()
    if !hasExternalDisplay && !needsDownload {
      yoloView.resetLayers()
      yoloView.setInferenceFlag(ok: false)
    }

    resetDownloadProgress()
    setLoadingState(true, showOverlay: true, canCancel: needsDownload)
    currentLoadingEntry = entry
    currentLoadingTask = task

    let yoloTask = appTasks.first(where: { $0.name == task })?.yoloTask ?? .detect

    if let url = entry.installedURL {
      loadModelAndSetToYOLOView(at: url, yoloTask: yoloTask, displayName: entry.displayName)
    } else if entry.isLocalBundle {
      DispatchQueue.global().async { [weak self] in
        guard let self = self else { return }

        guard let folderURL = appTasks.first(where: { $0.name == task })?.folder,
          let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
        else {
          DispatchQueue.main.async { [weak self] in
            self?.finishLoadingModel(success: false, modelName: entry.displayName)
          }
          return
        }

        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        DispatchQueue.main.async { [weak self] in
          guard let self = self else { return }
          self.downloadProgressLabel.isHidden = false
          self.downloadProgressLabel.text = "Loading \(entry.displayName)"

          // External display path: notify it instead of loading into the local YOLOView.
          if self.hasExternalScreen() {
            self.finishLoadingModel(success: true, modelName: entry.displayName)
          } else {
            self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) {
              [weak self] result in
              DispatchQueue.main.async {
                switch result {
                case .success():
                  self?.finishLoadingModel(success: true, modelName: entry.displayName)
                case .failure(let error):
                  print(error)
                  self?.finishLoadingModel(success: false, modelName: entry.displayName)
                }
              }
            }
          }
        }
      }
    } else {
      let key = entry.identifier  // e.g. "yolo26n", "yolo26m-seg"

      if ModelCacheManager.shared.isModelDownloaded(key: key) {
        loadModelAndSetToYOLOView(
          at: ModelCacheManager.shared.modelURL(for: key), yoloTask: yoloTask,
          displayName: entry.displayName)
      } else {
        guard let remoteURL = entry.remoteURL else {
          self.finishLoadingModel(success: false, modelName: entry.displayName)
          return
        }

        self.downloadProgressView.progress = 0.0
        self.downloadProgressView.isHidden = false
        self.downloadProgressLabel.isHidden = false

        // Show the initial download message with a properly formatted model name.
        self.downloadProgressLabel.text = "Downloading \(processString(entry.displayName))"

        let localZipFileName = remoteURL.lastPathComponent  // e.g. "yolo26n.mlpackage.zip"

        ModelCacheManager.shared.loadModel(
          from: localZipFileName,
          remoteURL: remoteURL,
          key: key
        ) { [weak self] success, loadedKey in
          guard let self = self else { return }
          if !success {
            self.finishLoadingModel(success: false, modelName: entry.displayName)
            return
          }
          self.loadModelAndSetToYOLOView(
            at: ModelCacheManager.shared.modelURL(for: loadedKey),
            yoloTask: yoloTask,
            displayName: entry.displayName)
        }
      }
    }
  }

  private func loadModelAndSetToYOLOView(
    at localModelURL: URL, yoloTask: YOLOTask, displayName: String
  ) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"

      // External display path: notify it instead of loading into the local YOLOView.
      if self.hasExternalScreen() {
        self.finishLoadingModel(success: true, modelName: displayName)
      } else {
        self.yoloView.resetLayers()
        self.yoloView.setInferenceFlag(ok: false)
        self.yoloView.setModel(modelPathOrName: localModelURL.path, task: yoloTask) {
          [weak self] result in
          DispatchQueue.main.async {
            switch result {
            case .success():
              self?.finishLoadingModel(success: true, modelName: displayName)
            case .failure(let error):
              print(error)
              self?.finishLoadingModel(success: false, modelName: displayName)
            }
          }
        }
      }
    }
  }

  private func resetDownloadProgress() {
    downloadProgressView.progress = 0.0
    downloadProgressLabel.text = ""
    [downloadProgressView, downloadProgressLabel, cancelLoadingButton].forEach {
      $0.isHidden = true
    }
  }

  @objc private func cancelLoadingModel() {
    guard isLoadingModel, let entry = currentLoadingEntry else { return }

    if !entry.isLocalBundle {
      ModelDownloadManager.shared.cancelDownload(key: entry.identifier)
    }

    setLoadingState(false)
    isLoadingModel = false
    currentLoadingEntry = nil
    currentLoadingTask = nil
    resetDownloadProgress()
    restoreCurrentModelSelection()

    if !hasExternalScreen(), !currentModelName.isEmpty {
      yoloView.setInferenceFlag(ok: true)
    }
    if currentModelName.isEmpty { loadSavedModelOrFirstOfficial() }
  }

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      self.setLoadingState(false)
      self.isLoadingModel = false
      self.resetDownloadProgress()
      let loadedTask = self.currentLoadingTask ?? self.currentTask
      let hasExternalDisplay = self.hasExternalScreen()
      let keptCurrentModel = !success && !self.currentModelName.isEmpty

      if success {
        let previousTask = self.currentTask
        self.currentTask = loadedTask
        if previousTask != loadedTask {
          NotificationCenter.default.post(
            name: .taskDidChange,
            object: nil,
            userInfo: ["task": loadedTask]
          )
        }
        self.currentModelName =
          self.currentLoadingEntry?.installedURL == nil ? processString(modelName) : modelName
        self.currentModelIdentifier = self.currentLoadingEntry?.identifier ?? ""
        UserDefaults.standard.set(
          self.currentLoadingEntry?.installedURL == nil ? nil : self.currentModelIdentifier,
          forKey: "selectedCustomModel")

        self.refreshModelButton()
      } else {
        self.restoreCurrentModelSelection()
      }

      if success && hasExternalDisplay {
        let yoloTask = appTasks.first(where: { $0.name == loadedTask })?.yoloTask ?? .detect
        var fullModelPath = ""

        if let entry = self.currentLoadingEntry {
          if let url = entry.installedURL {
            fullModelPath = url.path
          } else if entry.isLocalBundle {
            if let folderURL = appTasks.first(where: { $0.name == loadedTask })?.folder,
              let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
            {
              fullModelPath = folderPathURL.appendingPathComponent(entry.identifier).path
            }
          } else {
            let localModelURL = ModelCacheManager.shared.modelURL(for: entry.identifier)
            fullModelPath =
              FileManager.default.fileExists(atPath: localModelURL.path)
              ? localModelURL.path : ""
          }
        }

        if !fullModelPath.isEmpty {
          ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
          self.checkAndNotifyExternalDisplayIfReady()
        }
      }

      // Only toggle the local YOLOView inference flag when no external display is active.
      if !hasExternalDisplay {
        self.yoloView.setInferenceFlag(ok: success || keptCurrentModel)
      }

      if success {
        self.labelName.text = self.currentModelName
      }
      self.currentLoadingEntry = nil
      self.currentLoadingTask = nil
    }
  }

  private func setupModelActions() {
    var scanConfiguration = UIButton.Configuration.plain()
    scanConfiguration.image = UIImage(
      systemName: "qrcode.viewfinder",
      withConfiguration: yoloView.shareButton.currentImage?.symbolConfiguration)
    scanConfiguration.cornerStyle = .capsule
    scanButton.configuration = scanConfiguration
    scanButton.accessibilityLabel = "Scan QR"
    scanButton.addTarget(self, action: #selector(scanModelQRCode), for: .touchUpInside)
    yoloView.toolbar.addSubview(scanButton)

    // The task eyebrow and title move into one control above the performance line; it opens the picker.
    modelsButton.addTarget(self, action: #selector(showModelPicker), for: .touchUpInside)
    qrOverlay.cancelButton.addTarget(self, action: #selector(scanModelQRCode), for: .touchUpInside)
    [modelsButton, qrOverlay].forEach { $0.translatesAutoresizingMaskIntoConstraints = false }
    view.addSubview(modelsButton)
    yoloView.insertSubview(qrOverlay, belowSubview: yoloView.toolbar)
    NSLayoutConstraint.activate([
      modelsButton.topAnchor.constraint(
        equalTo: yoloView.layoutMarginsGuide.topAnchor, constant: -10),
      modelsButton.centerXAnchor.constraint(equalTo: yoloView.centerXAnchor),
      modelsButton.widthAnchor.constraint(lessThanOrEqualTo: yoloView.widthAnchor, constant: -40),
      labelFPS.topAnchor.constraint(equalTo: modelsButton.bottomAnchor, constant: 5),
      qrOverlay.topAnchor.constraint(equalTo: yoloView.topAnchor),
      qrOverlay.bottomAnchor.constraint(equalTo: yoloView.bottomAnchor),
      qrOverlay.leadingAnchor.constraint(equalTo: yoloView.leadingAnchor),
      qrOverlay.trailingAnchor.constraint(equalTo: yoloView.trailingAnchor),
    ])
    updateActivityState()
    refreshModelButton()
  }

  @objc private func showModelPicker() {
    let picker = ModelPickerController(
      custom: ModelCacheManager.shared.customModels,
      official: appTasks.compactMap { task in
        let names = Set((remoteModelsInfo[task.name] ?? []).map { $0.modelName.lowercased() })
        let entries = makeModelEntries(for: task.name).filter {
          names.contains($0.displayName.lowercased())
        }
        return entries.isEmpty ? nil : (task: task.name, entries: entries)
      }, selectedID: currentModelIdentifier,
      onScan: { [weak self] in self?.scanModelQRCode() },
      onHelp: {
        UIApplication.shared.open(
          URL(string: "https://docs.ultralytics.com/platform/train/models#export-model")!)
      },
      onSelect: { [weak self] entry in
        guard let self, !self.isLoadingModel, entry.identifier != self.currentModelIdentifier else {
          return
        }
        let task = appTasks.first { String(describing: $0.yoloTask) == entry.task }!.name
        self.showModelEntries(for: task)
        self.loadModel(entry: entry, forTask: task)
      },
      onRemove: { [weak self] entry in
        guard let self, !self.isLoadingModel else { return }
        do {
          try ModelCacheManager.shared.removeCustomModel(entry)
          if self.currentModelIdentifier == entry.identifier {
            self.reloadModelEntriesAndLoadFirst(for: self.currentTask)
          }
        } catch { self.showImportError(error) }
      })
    picker.popoverPresentationController?.sourceItem = modelsButton
    present(picker, animated: true)
  }

  private func refreshModelButton() {
    let isCustom = ModelCacheManager.shared.customModels.contains {
      $0.identifier == currentModelIdentifier
    }
    let task = "\(currentTask)\(isCustom ? " · Custom" : "")"
    modelsButton.taskLabel.attributedText = NSAttributedString(
      string: currentTask.isEmpty ? " " : task.uppercased(), attributes: [.kern: 1])
    modelsButton.accessibilityLabel =
      currentModelName.isEmpty
      ? "Models" : "Models, \(currentModelName), \(currentTask)\(isCustom ? ", Custom" : "")"
  }

  @objc private func scanModelQRCode() {
    guard !isLoadingModel else { return }
    if let qrCamera {
      qrCamera.cancelQRCodeScan()
      return
    }
    let externalCamera = UIApplication.shared.connectedScenes
      .compactMap { $0 as? UIWindowScene }.flatMap { $0.windows }
      .compactMap { $0.rootViewController as? ExternalViewController }.first?.yoloView
    let camera = externalCamera ?? yoloView!
    qrCamera = camera
    camera.showOverlays = false
    camera.scanQRCode { [weak self, weak camera] result in
      guard let self, let camera else { return }
      self.qrCamera = nil
      camera.showOverlays = true
      switch result {
      case .success(let value):
        if let value, let url = URL(string: value) { self.importModel(from: url) }
      case .failure(let error): self.showImportError(error)
      }
    }
  }

  func importModel(from url: URL) {
    qrCamera?.cancelQRCodeScan()
    guard !isLoadingModel else { return }
    if presentedViewController != nil {
      dismiss(animated: true) { [weak self] in self?.importModel(from: url) }
      return
    }
    isLoadingModel = true
    setLoadingState(true, showOverlay: true)
    Task { @MainActor in
      do {
        let model = try await MobileModelImport.resolve(url)
        let task = appTasks.first { String(describing: $0.yoloTask) == model.task }!
        isLoadingModel = false
        setLoadingState(false)
        if let installed = ModelCacheManager.shared.customModels.first(where: {
          $0.identifier == model.exportId
        }) {
          showModelEntries(for: task.name)
          loadModel(entry: installed, forTask: task.name)
          return
        }
        let alert = UIAlertController(
          title: model.name,
          message:
            "\(task.name) · \(ByteCountFormatter.string(fromByteCount: model.size, countStyle: .file))\nDownload for offline use?",
          preferredStyle: .alert)
        alert.addAction(
          UIAlertAction(title: "Cancel", style: .cancel) { [weak self] _ in
            guard let self, self.currentModelName.isEmpty else { return }
            self.loadSavedModelOrFirstOfficial()
          })
        alert.addAction(
          UIAlertAction(title: "Download", style: .default) { [weak self] _ in
            self?.installImportedModel(model)
          })
        present(alert, animated: true)
      } catch {
        isLoadingModel = false
        setLoadingState(false)
        showImportError(error)
      }
    }
  }

  private func installImportedModel(_ model: MobileModelImport) {
    guard !isLoadingModel else { return }
    isLoadingModel = true
    currentLoadingEntry = model.entry
    setLoadingState(true, showOverlay: true, canCancel: true)
    downloadProgressView.isHidden = false
    ModelDownloadManager.shared.startDownload(
      url: model.downloadUrl, fileName: "\(model.exportId).zip", key: model.exportId,
      modelImport: model
    ) { [weak self] success, _ in
      DispatchQueue.main.async {
        guard let self else { return }
        self.isLoadingModel = false
        self.currentLoadingEntry = nil
        self.setLoadingState(false)
        self.resetDownloadProgress()
        guard success else {
          self.showImportError(
            NSError(
              domain: "ModelImport", code: 3,
              userInfo: [
                NSLocalizedDescriptionKey:
                  "Download failed. Check your connection and scan a new code."
              ]))
          return
        }
        do {
          try ModelCacheManager.shared.saveCustomModel(model.entry)
          let task = appTasks.first { String(describing: $0.yoloTask) == model.task }!
          self.showModelEntries(for: task.name)
          self.loadModel(entry: model.entry, forTask: task.name)
        } catch { self.showImportError(error) }
      }
    }
  }

  private func showImportError(_ error: Error) {
    let alert = UIAlertController(
      title: "Model import failed", message: error.localizedDescription, preferredStyle: .alert)
    alert.addAction(
      UIAlertAction(title: "OK", style: .cancel) { [weak self] _ in
        guard let self, self.currentModelName.isEmpty else { return }
        self.loadSavedModelOrFirstOfficial()
      })
    present(alert, animated: true)
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: Constants.logoURL) {
      UIApplication.shared.open(link)
    }
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    loadingOverlayView?.frame = view.bounds
    alignWithThresholdSliders()
  }

  /// Keeps the logo beside the sliders, near the content so screenshots keep it, and the version on the zoom row.
  private func alignWithThresholdSliders() {
    guard logoImage != nil, yoloView != nil else { return }
    // YOLOView lays out after this controller's pass; read its frames for the new bounds, not the previous ones.
    yoloView.layoutIfNeeded()
    let sliderMidY = (yoloView.sliderConf.frame.midY + yoloView.sliderIoU.frame.midY) / 2
    guard sliderMidY.isFinite, sliderMidY > 0 else { return }

    logoImage.center = CGPoint(x: logoImage.center.x, y: sliderMidY + 6)
    labelVersion.frame.origin = CGPoint(
      x: yoloView.sliderConf.frame.minX,
      y: yoloView.labelZoom.frame.midY - labelVersion.frame.height / 2)
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
    // Forward threshold values to the external display (no-op if none is connected).
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
  }

}

/// Task eyebrow above the model title and its chevron; the whole block opens the model picker.
private final class ModelHeaderControl: UIControl {
  let taskLabel = UILabel()
  private let chevron = UIImageView(
    image: UIImage(
      systemName: "chevron.down",
      withConfiguration: UIImage.SymbolConfiguration(pointSize: 13, weight: .semibold)))

  init(title: UILabel) {
    super.init(frame: .zero)
    taskLabel.font = UIFontMetrics(forTextStyle: .caption1).scaledFont(
      for: .systemFont(ofSize: 12, weight: .semibold))
    taskLabel.adjustsFontForContentSizeCategory = true
    taskLabel.textColor = UIColor.white.withAlphaComponent(0.62)
    chevron.tintColor = UIColor.white.withAlphaComponent(0.62)
    chevron.setContentCompressionResistancePriority(.required, for: .horizontal)
    title.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
    let row = UIStackView(arrangedSubviews: [title, chevron])
    row.spacing = 6
    row.alignment = .center
    let stack = UIStackView(arrangedSubviews: [taskLabel, row])
    stack.axis = .vertical
    stack.alignment = .center
    stack.isUserInteractionEnabled = false
    stack.translatesAutoresizingMaskIntoConstraints = false
    addSubview(stack)
    NSLayoutConstraint.activate([
      stack.topAnchor.constraint(equalTo: topAnchor),
      stack.bottomAnchor.constraint(equalTo: bottomAnchor),
      stack.leadingAnchor.constraint(equalTo: leadingAnchor),
      stack.trailingAnchor.constraint(equalTo: trailingAnchor),
    ])
    isAccessibilityElement = true
    accessibilityTraits = .button
  }

  required init?(coder: NSCoder) { return nil }

  override var isHighlighted: Bool { didSet { alpha = isHighlighted ? 0.5 : 1 } }

  override var isEnabled: Bool {
    didSet {
      chevron.alpha = isEnabled ? 1 : 0.35
      accessibilityTraits = isEnabled ? .button : [.button, .notEnabled]
    }
  }
}

/// Dims the running camera around a viewfinder and says which Platform QR code to scan. Touches outside Cancel pass
/// through to the camera, so pinch-to-zoom keeps working while scanning.
private final class QRScanOverlayView: UIView {
  let cancelButton = UIButton(type: .system)
  private let titleLabel = UILabel()
  private let messageLabel = UILabel()
  private let scrim = CAShapeLayer()
  private let corners = CAShapeLayer()

  override init(frame: CGRect) {
    super.init(frame: frame)
    scrim.fillRule = .evenOdd
    scrim.fillColor = UIColor.black.withAlphaComponent(0.56).cgColor
    corners.fillColor = nil
    corners.strokeColor = UIColor.white.cgColor
    corners.lineWidth = 4
    corners.lineCap = .round
    layer.addSublayer(scrim)
    layer.addSublayer(corners)

    titleLabel.text = "Scan a model QR code"
    titleLabel.font = UIFontMetrics(forTextStyle: .title3).scaledFont(
      for: .systemFont(ofSize: 20, weight: .semibold))
    titleLabel.textColor = .white
    messageLabel.text = "In Ultralytics Platform, open an iPhone export and choose Open on phone."
    messageLabel.font = .preferredFont(forTextStyle: .subheadline)
    messageLabel.textColor = UIColor.white.withAlphaComponent(0.78)
    messageLabel.numberOfLines = 0
    [titleLabel, messageLabel].forEach {
      $0.textAlignment = .center
      $0.adjustsFontForContentSizeCategory = true
      addSubview($0)
    }

    var configuration = UIButton.Configuration.filled()
    configuration.title = "Cancel"
    configuration.baseForegroundColor = .white
    configuration.baseBackgroundColor = UIColor(
      red: 118 / 255, green: 118 / 255, blue: 128 / 255, alpha: 0.32)
    configuration.cornerStyle = .capsule
    configuration.contentInsets = NSDirectionalEdgeInsets(
      top: 12, leading: 22, bottom: 12, trailing: 22)
    configuration.titleTextAttributesTransformer = UIConfigurationTextAttributesTransformer {
      var attributes = $0
      attributes.font = UIFontMetrics(forTextStyle: .subheadline).scaledFont(
        for: .systemFont(ofSize: 15, weight: .semibold))
      return attributes
    }
    cancelButton.configuration = configuration
    addSubview(cancelButton)
  }

  required init?(coder: NSCoder) { return nil }

  override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {
    cancelButton.frame.contains(point)
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    // Portrait stacks the copy, viewfinder and Cancel; landscape puts the copy and Cancel beside the viewfinder.
    let wide = bounds.width > bounds.height
    let side = min(236, wide ? bounds.height * 0.5 : bounds.width * 0.6)
    let textWidth = min(310, wide ? bounds.width / 2 - 72 : bounds.width - 80)
    let fit = CGSize(width: textWidth, height: .greatestFiniteMagnitude)
    let titleHeight = titleLabel.sizeThatFits(fit).height
    let messageHeight = messageLabel.sizeThatFits(fit).height
    let button = cancelButton.intrinsicContentSize
    let textHeight = titleHeight + 6 + messageHeight
    let viewfinder: CGRect
    var textOrigin: CGPoint
    if wide {
      let left = (bounds.width - side - 32 - textWidth) / 2
      viewfinder = CGRect(x: left, y: bounds.midY - side / 2, width: side, height: side)
      textOrigin = CGPoint(
        x: viewfinder.maxX + 32, y: bounds.midY - (textHeight + 20 + button.height) / 2)
    } else {
      let top = bounds.midY - (textHeight + 24 + side + 20 + button.height) / 2
      textOrigin = CGPoint(x: bounds.midX - textWidth / 2, y: top)
      viewfinder = CGRect(
        x: bounds.midX - side / 2, y: top + textHeight + 24, width: side, height: side)
    }
    titleLabel.frame = CGRect(
      origin: textOrigin, size: CGSize(width: textWidth, height: titleHeight))
    textOrigin.y += titleHeight + 6
    messageLabel.frame = CGRect(
      origin: textOrigin, size: CGSize(width: textWidth, height: messageHeight))
    let buttonCenterX = wide ? textOrigin.x + textWidth / 2 : viewfinder.midX
    let buttonTop = wide ? textOrigin.y + messageHeight + 20 : viewfinder.maxY + 20
    cancelButton.frame = CGRect(
      x: buttonCenterX - button.width / 2, y: buttonTop, width: button.width, height: button.height)

    CATransaction.begin()
    CATransaction.setDisableActions(true)
    let cutout = UIBezierPath(rect: bounds)
    cutout.append(UIBezierPath(roundedRect: viewfinder, cornerRadius: 28))
    scrim.path = cutout.cgPath
    corners.path = Self.cornerPath(in: viewfinder.insetBy(dx: 2, dy: 2), radius: 26, arm: 44).cgPath
    CATransaction.commit()
  }

  private static func cornerPath(in r: CGRect, radius: CGFloat, arm: CGFloat) -> UIBezierPath {
    let path = UIBezierPath()
    path.move(to: CGPoint(x: r.minX, y: r.minY + arm))
    path.addArc(
      withCenter: CGPoint(x: r.minX + radius, y: r.minY + radius), radius: radius,
      startAngle: .pi, endAngle: 1.5 * .pi, clockwise: true)
    path.addLine(to: CGPoint(x: r.minX + arm, y: r.minY))
    path.move(to: CGPoint(x: r.maxX - arm, y: r.minY))
    path.addArc(
      withCenter: CGPoint(x: r.maxX - radius, y: r.minY + radius), radius: radius,
      startAngle: 1.5 * .pi, endAngle: 2 * .pi, clockwise: true)
    path.addLine(to: CGPoint(x: r.maxX, y: r.minY + arm))
    path.move(to: CGPoint(x: r.maxX, y: r.maxY - arm))
    path.addArc(
      withCenter: CGPoint(x: r.maxX - radius, y: r.maxY - radius), radius: radius,
      startAngle: 0, endAngle: 0.5 * .pi, clockwise: true)
    path.addLine(to: CGPoint(x: r.maxX - arm, y: r.maxY))
    path.move(to: CGPoint(x: r.minX + arm, y: r.maxY))
    path.addArc(
      withCenter: CGPoint(x: r.minX + radius, y: r.maxY - radius), radius: radius,
      startAngle: 0.5 * .pi, endAngle: .pi, clockwise: true)
    path.addLine(to: CGPoint(x: r.minX, y: r.maxY - arm))
    return path
  }
}

private final class ModelPickerController: UITableViewController,
  UIPopoverPresentationControllerDelegate
{
  private let custom: [ModelEntry]
  private let official: [(task: String, entries: [ModelEntry])]
  private let selectedID: String
  private let onScan: () -> Void
  private let onHelp: () -> Void
  private let onSelect: (ModelEntry) -> Void
  private let onRemove: (ModelEntry) -> Void

  init(
    custom: [ModelEntry], official: [(task: String, entries: [ModelEntry])], selectedID: String,
    onScan: @escaping () -> Void, onHelp: @escaping () -> Void,
    onSelect: @escaping (ModelEntry) -> Void, onRemove: @escaping (ModelEntry) -> Void
  ) {
    self.custom = custom
    self.official = official
    self.selectedID = selectedID
    self.onScan = onScan
    self.onHelp = onHelp
    self.onSelect = onSelect
    self.onRemove = onRemove
    super.init(style: .plain)
    tableView.backgroundColor = .clear
    tableView.separatorStyle = .none
    tableView.sectionHeaderTopPadding = 0
    modalPresentationStyle = .popover
    overrideUserInterfaceStyle = .dark
    popoverPresentationController?.delegate = self
    popoverPresentationController?.permittedArrowDirections = [.up, .down]
  }

  required init?(coder: NSCoder) { return nil }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    preferredContentSize = CGSize(width: 350, height: tableView.contentSize.height)
  }

  func adaptivePresentationStyle(
    for _: UIPresentationController, traitCollection _: UITraitCollection
  )
    -> UIModalPresentationStyle
  { .none }

  override func viewIsAppearing(_ animated: Bool) {
    super.viewIsAppearing(animated)
    tableView.layoutIfNeeded()
    let selected: IndexPath?
    if let row = custom.firstIndex(where: { $0.identifier == selectedID }) {
      selected = IndexPath(row: row + 1, section: 0)
    } else {
      selected = official.firstIndex(where: { $0.entries.contains { $0.identifier == selectedID } })
        .map { IndexPath(row: $0, section: 1) }
    }
    if let selected { tableView.scrollToRow(at: selected, at: .middle, animated: false) }
  }

  override func numberOfSections(in _: UITableView) -> Int { 2 }

  override func tableView(_: UITableView, numberOfRowsInSection section: Int) -> Int {
    section == 1 ? official.count : custom.count + 1
  }

  override func tableView(_: UITableView, titleForHeaderInSection section: Int) -> String? {
    section == 1 ? "Official" : "Custom"
  }

  override func tableView(_: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    indexPath.section == 1 ? 52 : UITableView.automaticDimension
  }

  override func tableView(_: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    let cell = UITableViewCell()
    cell.backgroundColor = .clear
    var content = UIListContentConfiguration.subtitleCell()
    content.textProperties.font = .systemFont(ofSize: 15, weight: .medium)
    content.textProperties.color = .white
    content.textProperties.numberOfLines = 1
    content.secondaryTextProperties.font = .systemFont(ofSize: 12)
    content.secondaryTextProperties.color = .white.withAlphaComponent(0.72)
    if indexPath.section == 1 {
      let group = official[indexPath.row]
      content.text = group.task
      let sizes = group.entries.map { String($0.displayName.split(separator: "-")[0].suffix(1)) }
      let control = UISegmentedControl(items: sizes)
      control.applyCameraStyle()
      // Segments share one title color, so sizes still to download draw their letter as a dimmed image.
      for (index, entry) in group.entries.enumerated()
      where !entry.isLocalBundle && !ModelCacheManager.shared.isModelDownloaded(key: entry.identifier) {
        let title = NSAttributedString(
          string: sizes[index],
          attributes: [
            .font: UIFont.systemFont(ofSize: 13, weight: .semibold),
            .foregroundColor: UIColor.white.withAlphaComponent(0.5),
          ])
        let image = UIGraphicsImageRenderer(size: title.size()).image { _ in title.draw(at: .zero) }
          .withRenderingMode(.alwaysOriginal)
        image.accessibilityLabel = "\(sizes[index]), not downloaded"
        control.setImage(image, forSegmentAt: index)
      }
      control.selectedSegmentIndex =
        group.entries.firstIndex { $0.identifier == selectedID }
        ?? UISegmentedControl.noSegment
      control.accessibilityLabel = "\(group.task) model size"
      control.frame = CGRect(x: 0, y: 0, width: 220, height: 34)
      control.addAction(
        UIAction { [weak self] action in
          guard let self, let control = action.sender as? UISegmentedControl else { return }
          let entry = group.entries[control.selectedSegmentIndex]
          self.dismiss(animated: true) { self.onSelect(entry) }
        }, for: .valueChanged)
      cell.accessoryView = control
      cell.selectionStyle = .none
    } else if indexPath.row == 0 {
      content.text = "Scan QR"
      content.image = UIImage(systemName: "qrcode.viewfinder")
      content.imageProperties.tintColor = .white
      let help = UIButton(
        type: .system,
        primaryAction: UIAction(image: UIImage(systemName: "questionmark.circle")) {
          [weak self] _ in
          guard let self else { return }
          self.dismiss(animated: true) { self.onHelp() }
        })
      help.frame = CGRect(x: 0, y: 0, width: 44, height: 44)
      help.tintColor = .white
      help.accessibilityLabel = "Platform export guide"
      cell.accessoryView = help
    } else {
      let entry = custom[indexPath.row - 1]
      content.text = entry.displayName
      content.image = UIImage(systemName: "checkmark")
      content.imageProperties.tintColor = entry.identifier == selectedID ? .white : .clear
      content.secondaryText = appTasks.first { String(describing: $0.yoloTask) == entry.task }?.name
      if entry.identifier == selectedID { cell.accessibilityTraits.insert(.selected) }
      let trash = UIButton(
        type: .system,
        primaryAction: UIAction(image: UIImage(systemName: "trash")) { [weak self] _ in
          guard let self else { return }
          self.dismiss(animated: true) { self.onRemove(entry) }
        })
      trash.frame = CGRect(x: 0, y: 0, width: 44, height: 44)
      trash.tintColor = .systemRed
      trash.accessibilityLabel = "Remove \(entry.displayName)"
      cell.accessoryView = trash
    }
    cell.contentConfiguration = content
    return cell
  }

  override func tableView(
    _ tableView: UITableView, willDisplayHeaderView view: UIView, forSection section: Int
  ) {
    var content = UIListContentConfiguration.plainHeader()
    content.text = self.tableView(tableView, titleForHeaderInSection: section)
    content.textProperties.font = .systemFont(ofSize: 12, weight: .semibold)
    content.textProperties.color = .white.withAlphaComponent(0.72)
    (view as? UITableViewHeaderFooterView)?.contentConfiguration = content
  }

  override func tableView(_: UITableView, didSelectRowAt indexPath: IndexPath) {
    guard indexPath.section == 0 else { return }
    if indexPath.row == 0 {
      dismiss(animated: true) { self.onScan() }
    } else {
      let entry = custom[indexPath.row - 1]
      dismiss(animated: true) { self.onSelect(entry) }
    }
  }
}

// MARK: - YOLOViewDelegate
extension ViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { [weak self] in
      self?.fpsText = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
      self?.renderPerformanceLabel()
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    let breakdownTotal = result.preMs + result.inferenceMs + result.postMs
    let text =
      breakdownTotal > 0
      ? String(
        format: "%.1f pre · %.1f inference · %.1f post", result.preMs, result.inferenceMs,
        result.postMs)
      : ""
    DispatchQueue.main.async { [weak self] in
      self?.breakdownText = text
      self?.renderPerformanceLabel()
    }
  }

  /// Renders the FPS line with the smaller stage-breakdown line beneath it in the single `labelFPS`,
  /// so the breakdown inherits the label's exact layout and show/hide behavior (e.g. camera switching).
  private func renderPerformanceLabel() {
    let rendered = NSMutableAttributedString(
      string: fpsText,
      attributes: [
        .font: UIFont.preferredFont(forTextStyle: .body), .foregroundColor: UIColor.white,
      ])
    if !breakdownText.isEmpty {
      rendered.append(
        NSAttributedString(
          string: "\n" + breakdownText,
          attributes: [
            .font: UIFont.preferredFont(forTextStyle: .caption1),
            .foregroundColor: UIColor.white.withAlphaComponent(0.7),
          ]))
    }
    labelFPS.attributedText = rendered
  }

}
