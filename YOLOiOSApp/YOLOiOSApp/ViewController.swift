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

// MARK: - ModelTableViewCell
class ModelTableViewCell: UITableViewCell {
  static let identifier = "ModelTableViewCell"

  override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
    super.init(style: .default, reuseIdentifier: reuseIdentifier)
    backgroundColor = .clear
    selectionStyle = .default
    textLabel?.textAlignment = .center
    textLabel?.font = .systemFont(ofSize: 14, weight: .medium)
    textLabel?.adjustsFontSizeToFitWidth = true
    textLabel?.minimumScaleFactor = 0.7
    selectedBackgroundView = {
      let v = UIView()
      v.backgroundColor = UIColor(white: 1.0, alpha: 0.3)
      v.layer.cornerRadius = 5
      return v
    }()
  }

  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  func configure(with modelName: String, isRemote: Bool, isDownloaded: Bool) {
    textLabel?.text = modelName
    textLabel?.textColor = (isRemote && !isDownloaded) ? .lightGray : .white
    imageView?.image =
      (isRemote && !isDownloaded)
      ? UIImage(
        systemName: "icloud.and.arrow.down",
        withConfiguration: UIImage.SymbolConfiguration(pointSize: 14)) : nil
    imageView?.tintColor = .white
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    selectedBackgroundView?.frame = bounds.insetBy(dx: 2, dy: 1)
  }
}

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate {

  // MARK: - External Display Support (Optional)
  // NOTE: The following orientation overrides are part of the OPTIONAL external display feature.
  // They can be safely removed if external display support is not needed.
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
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var focus: UIImageView!
  @IBOutlet weak var logoImage: UIImageView!

  let selection = UISelectionFeedbackGenerator()

  // Store current loading entry for external display notification (Optional feature)
  var currentLoadingEntry: ModelEntry?

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
    [view, modelTableView].forEach { $0.isUserInteractionEnabled = !loading }
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

  var currentTask: String = ""
  var currentModelName: String = ""

  private var isLoadingModel = false

  let modelTableView: UITableView = {
    let table = UITableView()
    table.isHidden = true
    table.layer.cornerRadius = 5  // Match corner radius of other elements
    table.clipsToBounds = true
    return table
  }()

  let tableViewBGView = UIView()

  var selectedIndexPath: IndexPath?

  override func viewDidLoad() {
    super.viewDidLoad()

    // Debug: Check model folders
    debugCheckModelFolders()

    // MARK: External Display Setup (Optional)
    // NOTE: The following external display setup is OPTIONAL and not required for core app functionality.
    // This code enhances the app for external monitor/TV connections but can be safely removed.
    // See ExternalDisplay/ directory and README for more information.

    // Setup external display notifications
    setupExternalDisplayNotifications()

    // Check for already connected external displays
    checkForExternalDisplays()

    // If external display is already connected, ensure YOLOView doesn't interfere
    if UIScreen.screens.count > 1 {
      print("External display already connected at startup - deferring camera init")
      yoloView.isHidden = true
    }

    // Setup segmented control and load models
    segmentedControl.removeAllSegments()
    tasks.enumerated().forEach { index, task in
      segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
      modelsForTask[task.name] = getModelFiles(in: task.folder)
    }

    if tasks.indices.contains(Constants.defaultTaskIndex) {
      segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
      currentTask = tasks[Constants.defaultTaskIndex].name
      
      // Load models and auto-select first one if no external display
      if UIScreen.screens.count == 1 {
        // No external display - load model and start inference
        reloadModelEntriesAndLoadFirst(for: currentTask)
      } else {
        // External display connected - stop main YOLOView
        print("External display detected at startup - stopping main YOLOView")
        yoloView.stop()
        yoloView.isHidden = true

        // Load model list for UI but don't auto-select
        currentModels = makeModelEntries(for: currentTask)
        modelTableView.reloadData()

        if !currentModels.isEmpty {
          modelTableView.isHidden = false
        }

        // Ensure camera is fully released
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
          print("Main YOLOView fully stopped, external display can now use camera")
        }
      }
    }

    setupTableView()

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
    modelTableView.isHidden = currentModels.isEmpty
    modelTableView.reloadData()

    if !currentModels.isEmpty {
      DispatchQueue.main.async {
        let firstIndex = IndexPath(row: 0, section: 0)
        self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
        self.selectedIndexPath = firstIndex
        self.loadModel(entry: self.currentModels[0], forTask: taskName)
      }
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
    isLoadingModel = true
    
    // Check if external display is connected
    let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay
    
    // Only reset YOLOView if no external display is connected
    if !hasExternalDisplay {
      yoloView.resetLayers()
      yoloView.setInferenceFlag(ok: false)
    }
    
    setLoadingState(true, showOverlay: true)
    resetDownloadProgress()

    print("Start loading model: \(entry.displayName)")
    print("  - displayName: \(entry.displayName)")
    print("  - identifier: \(entry.identifier)")
    print("  - isLocalBundle: \(entry.isLocalBundle)")
    print("  - task: \(task)")
    print("  - hasExternalDisplay: \(hasExternalDisplay)")

    // Store current entry for external display notification
    currentLoadingEntry = entry

    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect

    if entry.isLocalBundle {
      DispatchQueue.global().async { [weak self] in
        guard let self = self else { return }

        guard let folderURL = self.tasks.first(where: { $0.name == task })?.folder,
          let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
        else {
          DispatchQueue.main.async {
            self.finishLoadingModel(success: false, modelName: entry.displayName)
          }
          return
        }

        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        DispatchQueue.main.async {
          self.downloadProgressLabel.isHidden = false
          self.downloadProgressLabel.text = "Loading \(entry.displayName)"
          
          // Check if external display is connected
          let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay
          
          if hasExternalDisplay {
            // External display is connected - skip YOLOView loading, just notify external display
            print("External display connected - skipping main YOLOView model load")
            self.finishLoadingModel(success: true, modelName: entry.displayName)
          } else {
            // Normal model loading on main YOLOView
            self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
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
    } else {
      let key = entry.identifier  // "yolov8n", "yolov8m-seg", etc.

      if ModelCacheManager.shared.isModelDownloaded(key: key) {
        loadCachedModelAndSetToYOLOView(
          key: key, yoloTask: yoloTask, displayName: entry.displayName)
      } else {
        guard let remoteURL = entry.remoteURL else {
          self.finishLoadingModel(success: false, modelName: entry.displayName)
          return
        }

        self.downloadProgressView.progress = 0.0
        self.downloadProgressView.isHidden = false
        self.downloadProgressLabel.isHidden = false

        let localZipFileName = remoteURL.lastPathComponent  // ex. "yolov8n.mlpackage.zip"

        ModelCacheManager.shared.loadModel(
          from: localZipFileName,
          remoteURL: remoteURL,
          key: key
        ) { [weak self] mlModel, loadedKey in
          guard let self = self else { return }
          if mlModel == nil {
            self.finishLoadingModel(success: false, modelName: entry.displayName)
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

    DispatchQueue.main.async {
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"
      
      // Check if external display is connected
      let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay
      
      if hasExternalDisplay {
        // External display is connected - skip YOLOView loading, just notify external display
        print("External display connected - skipping main YOLOView cached model load")
        self.finishLoadingModel(success: true, modelName: displayName)
      } else {
        // Normal model loading on main YOLOView
        self.yoloView.setModel(modelPathOrName: localModelURL.path, task: yoloTask) { result in
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

  private func resetDownloadProgress() {
    downloadProgressView.progress = 0.0
    [downloadProgressView, downloadProgressLabel].forEach { $0.isHidden = true }
  }

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async {
      self.setLoadingState(false)
      self.isLoadingModel = false
      self.resetDownloadProgress()

      self.modelTableView.reloadData()

      // Notify external display of model change (Optional feature - can be safely removed)
      if success {
        // Update currentModelName
        self.currentModelName = processString(modelName)

        let yoloTask = self.tasks.first(where: { $0.name == self.currentTask })?.yoloTask ?? .detect

        // Determine the correct model path for external display
        var fullModelPath = ""

        // Use the stored entry from loadModel
        if let entry = self.currentLoadingEntry {
          if entry.isLocalBundle {
            // For local bundle models
            if let folderURL = self.tasks.first(where: { $0.name == self.currentTask })?.folder,
              let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
            {
              let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
              fullModelPath = modelURL.path
              print("📦 External display local model path: \(fullModelPath)")
            }
          } else {
            // For remote/downloaded models, we need to pass the identifier only
            // The external display will handle loading from cache
            fullModelPath = entry.identifier
            print("☁️ External display will load cached model: \(fullModelPath)")
            
            // Verify the cached model exists locally first
            let documentsDirectory = FileManager.default.urls(
              for: .documentDirectory, in: .userDomainMask)[0]
            let localModelURL =
              documentsDirectory
              .appendingPathComponent(entry.identifier)
              .appendingPathExtension("mlmodelc")
            
            if !FileManager.default.fileExists(atPath: localModelURL.path) {
              print("❌ Cached model not found at: \(localModelURL.path)")
              return
            }
          }
        }

        // Only notify if we have a valid path
        if !fullModelPath.isEmpty {
          ExternalDisplayManager.shared.notifyModelChange(task: yoloTask, modelName: fullModelPath)
          print("✅ Model loaded successfully and notified to external display: \(modelName)")

          // Also check if external display is waiting for initial model
          self.checkAndNotifyExternalDisplayIfReady()
        } else {
          print("❌ Could not determine model path for external display")
        }
      }

      if let ip = self.selectedIndexPath {
        self.modelTableView.selectRow(at: ip, animated: false, scrollPosition: .none)
      }

      // Check if external display is connected
      let hasExternalDisplay = UIScreen.screens.count > 1 || SceneDelegate.hasExternalDisplay
      
      // Only set inference flag on YOLOView if no external display
      if !hasExternalDisplay {
        self.yoloView.setInferenceFlag(ok: success)
      }

      if success {
        self.currentModelName = modelName
        self.labelName.text = processString(modelName)
      }
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
    selectedIndexPath = nil

    // Notify external display of task change immediately (Optional external display feature)
    NotificationCenter.default.post(
      name: .taskDidChange,
      object: nil,
      userInfo: ["task": newTask]
    )

    reloadModelEntriesAndLoadFirst(for: currentTask)
    updateTableViewBGFrame()
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: Constants.logoURL) {
      UIApplication.shared.open(link)
    }
  }

  private func setupTableView() {
    modelTableView.delegate = self
    modelTableView.dataSource = self
    modelTableView.register(
      ModelTableViewCell.self, forCellReuseIdentifier: ModelTableViewCell.identifier)
    modelTableView.backgroundColor = .clear
    modelTableView.separatorStyle = .none
    modelTableView.isScrollEnabled = false
    modelTableView.translatesAutoresizingMaskIntoConstraints = false
    tableViewBGView.backgroundColor = .darkGray.withAlphaComponent(0.3)
    tableViewBGView.layer.cornerRadius = 5
    tableViewBGView.clipsToBounds = true
    [tableViewBGView, modelTableView].forEach { yoloView.addSubview($0) }
    updateTableViewBGFrame()
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()

    // Layout model table view
    let isLandscape = view.bounds.width > view.bounds.height
    let tableViewWidth = view.bounds.width * (isLandscape ? 0.2 : 0.4)

    modelTableView.frame =
      isLandscape
      ? CGRect(x: segmentedControl.frame.maxX + 20, y: 20, width: tableViewWidth, height: 200)
      : CGRect(
        x: view.bounds.width - tableViewWidth - 8, y: segmentedControl.frame.maxY + 25,
        width: tableViewWidth, height: 200)

    updateTableViewBGFrame()
  }

  private func updateTableViewBGFrame() {
    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: CGFloat(currentModels.count * Int(Constants.tableRowHeight) + 2)
    )
  }

  @objc func shareButtonTapped() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] image in
      guard let self = self, let image = image else { return print("error capturing photo") }
      DispatchQueue.main.async {
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

    print("📊 Threshold changed - Conf: \(conf), IoU: \(iou), Max items: \(maxItems)")
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }

  private func debugCheckModelFolders() {
    print("\n🔍 DEBUG: Checking model folders...")
    let folders = ["DetectModels", "SegmentModels", "ClassifyModels", "PoseModels", "OBBModels"]

    for folder in folders {
      if let folderURL = Bundle.main.url(forResource: folder, withExtension: nil) {
        print("✅ \(folder) found at: \(folderURL.path)")

        do {
          let files = try FileManager.default.contentsOfDirectory(
            at: folderURL, includingPropertiesForKeys: nil)
          let models = files.filter {
            $0.pathExtension == "mlmodel" || $0.pathExtension == "mlpackage"
          }
          print("   📦 Models: \(models.map { $0.lastPathComponent })")
        } catch {
          print("   ❌ Error reading folder: \(error)")
        }
      } else {
        print("❌ \(folder) NOT FOUND in bundle")
      }
    }
    print("\n")
  }

}

// MARK: - UITableViewDataSource, UITableViewDelegate
extension ViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    currentModels.count
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    Constants.tableRowHeight
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    let cell =
      tableView.dequeueReusableCell(withIdentifier: ModelTableViewCell.identifier, for: indexPath)
      as! ModelTableViewCell
    let entry = currentModels[indexPath.row]

    // Check if the model is remote and not yet downloaded
    let isDownloaded =
      entry.isRemote ? ModelCacheManager.shared.isModelDownloaded(key: entry.identifier) : true

    // Format model name using the processString function
    let formattedName = processString(entry.displayName)

    // Configure the cell
    cell.configure(with: formattedName, isRemote: entry.isRemote, isDownloaded: isDownloaded)

    return cell
  }

  func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
    selection.selectionChanged()

    selectedIndexPath = indexPath
    let selectedEntry = currentModels[indexPath.row]

    loadModel(entry: selectedEntry, forTask: currentTask)
  }

}

// MARK: - YOLOViewDelegate
extension ViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async {
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
      self.labelFPS.textColor = .white
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    DispatchQueue.main.async {
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