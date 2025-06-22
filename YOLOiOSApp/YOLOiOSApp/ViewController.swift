// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO app, providing the main user interface for model selection and visualization.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController serves as the primary interface for users to interact with YOLO models.
//  It provides the ability to select different models, tasks (detection, segmentation, classification, etc.),
//  and visualize results in real-time. The controller manages the loading of local and remote models,
//  handles UI updates during model loading and inference, and provides functionality for capturing
//  and sharing detection results. Advanced features include screen recording, model download progress
//  tracking, and adaptive UI layout for different device orientations.

import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import ReplayKit
import UIKit
import YOLO

// Definition of custom table view cell
class ModelTableViewCell: UITableViewCell {
  static let identifier = "ModelTableViewCell"

  private let modelNameLabel: UILabel = {
    let label = UILabel()
    label.textAlignment = .center
    label.font = UIFont.systemFont(ofSize: 14, weight: .medium)
    label.translatesAutoresizingMaskIntoConstraints = false
    // Configure auto text size adjustment for long content
    label.adjustsFontSizeToFitWidth = true
    label.minimumScaleFactor = 0.7  // Scale down to 70% minimum
    label.lineBreakMode = .byClipping  // Clip text instead of using ellipsis
    return label
  }()

  private let downloadIconImageView: UIImageView = {
    let imageView = UIImageView(image: UIImage(systemName: "icloud.and.arrow.down"))
    imageView.tintColor = .white
    imageView.contentMode = .scaleAspectFit
    imageView.translatesAutoresizingMaskIntoConstraints = false
    imageView.isHidden = true
    return imageView
  }()

  override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
    super.init(style: style, reuseIdentifier: reuseIdentifier)
    setupUI()
  }

  required init?(coder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }

  private func setupUI() {
    backgroundColor = .clear
    selectionStyle = .default

    contentView.addSubview(modelNameLabel)
    contentView.addSubview(downloadIconImageView)

    NSLayoutConstraint.activate([
      // Center the label
      modelNameLabel.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),  // Add center X alignment
      modelNameLabel.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
      // Set margin from the leading edge
      modelNameLabel.leadingAnchor.constraint(
        greaterThanOrEqualTo: contentView.leadingAnchor, constant: 8),
      // Ensure margin between label and download icon
      modelNameLabel.trailingAnchor.constraint(
        lessThanOrEqualTo: downloadIconImageView.leadingAnchor, constant: -4),

      // Position download icon at the trailing edge
      downloadIconImageView.trailingAnchor.constraint(
        equalTo: contentView.trailingAnchor, constant: -4),
      downloadIconImageView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
      downloadIconImageView.widthAnchor.constraint(equalToConstant: 16),  // Slightly smaller to save space
      downloadIconImageView.heightAnchor.constraint(equalToConstant: 16),
    ])

    // Configure the background view for selection state
    let selectedBGView = UIView()
    selectedBGView.backgroundColor = UIColor(white: 1.0, alpha: 0.3)
    selectedBGView.layer.cornerRadius = 5  // More gentle corner radius
    selectedBGView.layer.masksToBounds = true
    selectedBackgroundView = selectedBGView
  }

  // Method to configure the cell
  func configure(with modelName: String, isRemote: Bool, isDownloaded: Bool) {
    modelNameLabel.text = modelName

    // Show download icon only for remote models that are not yet downloaded
    let showDownloadIcon = isRemote && !isDownloaded
    downloadIconImageView.isHidden = !showDownloadIcon

    // Adjust text priorities based on icon visibility
    if showDownloadIcon {
      // Keep center alignment when icon is visible
      modelNameLabel.textAlignment = .center
      // Adjust width to accommodate icon space
      modelNameLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
      // Set priority to ensure center alignment is respected
      modelNameLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
    } else {
      // Keep center alignment when no icon is present
      modelNameLabel.textAlignment = .center
      // Expand to full width
      modelNameLabel.setContentCompressionResistancePriority(.defaultHigh, for: .horizontal)
      // Prioritize center alignment
      modelNameLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
    }
  }

  override func layoutSubviews() {
    super.layoutSubviews()

    // Adjust selection background view size - reduce margins to bring frame closer to table view
    if let selectedBGView = selectedBackgroundView {
      selectedBGView.frame = bounds.insetBy(dx: 2, dy: 1)
    }

    // Final label adjustments - ensure label displays correctly after layout
    let iconSpace = downloadIconImageView.isHidden ? 0 : 20  // Consider icon space if visible
    let availableWidth = bounds.width - 16 - CGFloat(iconSpace)  // Left/right margins(16) + icon space

    // Set maximum label width to ensure center alignment works properly
    modelNameLabel.preferredMaxLayoutWidth = availableWidth

    // Fine-tune frame to enforce center alignment
    let labelFrame = modelNameLabel.frame
    if downloadIconImageView.isHidden {
      // Center completely when no icon is present
      modelNameLabel.center.x = bounds.width / 2
    } else {
      // Keep text centered in the cell even when download icon is visible
      modelNameLabel.center.x = bounds.width / 2
    }
  }
}

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate, ModelDropdownViewDelegate {

  @IBOutlet weak var yoloView: YOLOView!
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var forcus: UIImageView!
  @IBOutlet weak var logoImage: UIImageView!
  
  // New UI Components
  private let statusMetricBar = StatusMetricBar()
  private let cameraPreviewContainer = UIView()
  private let taskTabStrip = TaskTabStrip()
  private let shutterBar = ShutterBar()
  private let rightSideToolBar = RightSideToolBar()
  private let parameterEditView = ParameterEditView()
  private let modelDropdown = ModelDropdownView()
  
  // UI State
  private var isNewUIActive = true // Toggle for new/old UI
  private var currentThresholds: [String: Float] = [
    "confidence": 0.25,
    "iou": 0.45,
    "itemsMax": 15,
    "lineThickness": 2.0
  ]

  var shareButton = UIButton()
  var recordButton = UIButton()
  let selection = UISelectionFeedbackGenerator()
  var firstLoad = true

  private let downloadProgressView: UIProgressView = {
    let pv = UIProgressView(progressViewStyle: .default)
    pv.progress = 0.0
    pv.isHidden = true
    return pv
  }()

  private let downloadProgressLabel: UILabel = {
    let label = UILabel()
    label.text = ""
    label.textAlignment = .center
    label.textColor = .systemGray
    label.font = UIFont.systemFont(ofSize: 14)
    label.isHidden = true
    return label
  }()

  private var loadingOverlayView: UIView?

  func showLoadingOverlay() {
    guard loadingOverlayView == nil else { return }
    let overlay = UIView(frame: view.bounds)
    overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)

    view.addSubview(overlay)
    loadingOverlayView = overlay
    view.bringSubviewToFront(downloadProgressView)
    view.bringSubviewToFront(downloadProgressLabel)

    view.isUserInteractionEnabled = false
  }

  func hideLoadingOverlay() {
    loadingOverlayView?.removeFromSuperview()
    loadingOverlayView = nil
    view.isUserInteractionEnabled = true
  }

  private let tasks: [(name: String, folder: String)] = [
    ("Classify", "ClassifyModels"),  // index 0
    ("Segment", "SegmentModels"),  // index 1
    ("Detect", "DetectModels"),  // index 2
    ("Pose", "PoseModels"),  // index 3
    ("OBB", "OBBModels"),  // index 4
  ]

  private var modelsForTask: [String: [String]] = [:]

  private var currentModels: [ModelEntry] = []

  private var currentTask: String = ""
  private var currentModelName: String = ""

  private var isLoadingModel = false

  private let modelTableView: UITableView = {
    let table = UITableView()
    table.isHidden = true
    table.layer.cornerRadius = 5  // 他の要素のcorner radiusに合わせる
    table.clipsToBounds = true
    return table
  }()

  private let tableViewBGView = UIView()

  private var selectedIndexPath: IndexPath?

  override func viewDidLoad() {
    super.viewDidLoad()

    setupTaskSegmentedControl()
    loadModelsForAllTasks()

    if tasks.indices.contains(2) {
      segmentedControl.selectedSegmentIndex = 2
      currentTask = tasks[2].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    setupTableView()
    setupButtons()

    yoloView.delegate = self
    yoloView.labelName.isHidden = true
    yoloView.labelFPS.isHidden = true

    // ラベルのテキスト色を白色に強制設定
    labelName.textColor = .white
    labelFPS.textColor = .white
    labelVersion.textColor = .white

    // ダークモード/ライトモードの切り替えに影響されないようにスタイル設定
    labelName.overrideUserInterfaceStyle = .dark
    labelFPS.overrideUserInterfaceStyle = .dark
    labelVersion.overrideUserInterfaceStyle = .dark
    
    // Setup new UI if active
    if isNewUIActive {
      setupNewUI()
      hideOldUI()
    }

    downloadProgressView.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(downloadProgressView)

    downloadProgressLabel.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(downloadProgressLabel)

    NSLayoutConstraint.activate([
      downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      downloadProgressView.topAnchor.constraint(
        equalTo: activityIndicator.bottomAnchor, constant: 8),
      downloadProgressView.widthAnchor.constraint(equalToConstant: 200),
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

    // Force text color to white whenever the screen appears
    enforceWhiteTextColor()

    // Override system appearance mode setting to ensure consistent styling
    view.overrideUserInterfaceStyle = .dark
  }

  // Called when trait collection changes (dark mode/light mode)
  override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
    super.traitCollectionDidChange(previousTraitCollection)

    // Maintain white text color even when appearance mode changes
    enforceWhiteTextColor()
  }

  // Common method to set label text color to white
  private func enforceWhiteTextColor() {
    labelName.textColor = .white
    labelFPS.textColor = .white
    labelVersion.textColor = .white
  }

  private func setupTaskSegmentedControl() {
    segmentedControl.removeAllSegments()
    for (index, taskInfo) in tasks.enumerated() {
      segmentedControl.insertSegment(withTitle: taskInfo.name, at: index, animated: false)
    }
  }

  private func loadModelsForAllTasks() {
    for taskInfo in tasks {
      let taskName = taskInfo.name
      let folderName = taskInfo.folder
      let modelFiles = getModelFiles(in: folderName)
      modelsForTask[taskName] = modelFiles
    }
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil) else {
      return []
    }
    do {
      let fileURLs = try FileManager.default.contentsOfDirectory(
        at: folderURL,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
      )
      let modelFiles =
        fileURLs
        .filter { $0.pathExtension == "mlmodel" || $0.pathExtension == "mlpackage" }
        .map { $0.lastPathComponent }

      if folderName == "DetectModels" {
        return reorderDetectionModels(modelFiles)
      } else {
        return modelFiles.sorted()
      }

    } catch {
      print("Error reading contents of folder \(folderName): \(error)")
      return []
    }
  }

  private func reorderDetectionModels(_ fileNames: [String]) -> [String] {
    let officialOrder: [Character: Int] = ["n": 0, "m": 1, "s": 2, "l": 3, "x": 4]

    var customModels: [String] = []
    var officialModels: [String] = []

    for fileName in fileNames {
      let baseName = (fileName as NSString).deletingPathExtension.lowercased()

      if baseName.hasPrefix("yolo"),
        let lastChar = baseName.last,
        officialOrder.keys.contains(lastChar)
      {
        officialModels.append(fileName)
      } else {
        customModels.append(fileName)
      }
    }

    customModels.sort { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }

    officialModels.sort { fileA, fileB in
      let baseA = (fileA as NSString).deletingPathExtension.lowercased()
      let baseB = (fileB as NSString).deletingPathExtension.lowercased()
      guard let lastA = baseA.last, let lastB = baseB.last,
        let indexA = officialOrder[lastA], let indexB = officialOrder[lastB]
      else {
        return baseA < baseB
      }
      return indexA < indexB
    }

    return customModels + officialModels
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    currentModels = makeModelEntries(for: taskName)

    if !currentModels.isEmpty {
      modelTableView.isHidden = false
      modelTableView.reloadData()

      DispatchQueue.main.async {
        let firstIndex = IndexPath(row: 0, section: 0)
        self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
        self.selectedIndexPath = firstIndex
        let firstModel = self.currentModels[0]
        self.loadModel(entry: firstModel, forTask: taskName)
      }
    } else {
      print("No models found for task: \(taskName)")
      modelTableView.isHidden = true
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

    let remoteList = remoteModelsInfo[taskName] ?? []
    let remoteEntries = remoteList.map { (modelName, url) -> ModelEntry in
      ModelEntry(
        displayName: modelName,
        identifier: modelName,
        isLocalBundle: false,
        isRemote: true,
        remoteURL: url
      )
    }

    return localEntries + remoteEntries
  }

  private func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else {
      print("Model is already loading. Please wait.")
      return
    }
    isLoadingModel = true
    yoloView.resetLayers()
    if !firstLoad {
      showLoadingOverlay()
      yoloView.setInferenceFlag(ok: false)
    } else {
      firstLoad = false
    }

    self.activityIndicator.startAnimating()
    self.downloadProgressView.progress = 0.0
    self.downloadProgressView.isHidden = true
    self.downloadProgressLabel.isHidden = true
    self.view.isUserInteractionEnabled = false
    self.modelTableView.isUserInteractionEnabled = false

    print("Start loading model: \(entry.displayName)")

    if entry.isLocalBundle {
      DispatchQueue.global().async { [weak self] in
        guard let self = self else { return }
        let yoloTask = self.convertTaskNameToYOLOTask(task)

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
    } else {
      let yoloTask = self.convertTaskNameToYOLOTask(task)

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
    let localModelURL = ModelCacheManager.shared.getDocumentsDirectory()
      .appendingPathComponent(key)
      .appendingPathExtension("mlmodelc")

    DispatchQueue.main.async {
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"
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

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async {
      self.activityIndicator.stopAnimating()
      self.downloadProgressView.isHidden = true

      self.downloadProgressLabel.isHidden = true
      //            self.downloadProgressLabel.isHidden = false
      //            self.downloadProgressLabel.text = "Loading \(modelName)"

      self.view.isUserInteractionEnabled = true
      self.modelTableView.isUserInteractionEnabled = true
      self.isLoadingModel = false

      self.modelTableView.reloadData()

      if let ip = self.selectedIndexPath {
        self.modelTableView.selectRow(at: ip, animated: false, scrollPosition: .none)
      }
      if !self.firstLoad {
        self.hideLoadingOverlay()
      }
      self.yoloView.setInferenceFlag(ok: true)

      if success {
        print("Finished loading model: \(modelName)")
        self.currentModelName = modelName
        DispatchQueue.main.async {
          self.labelName.text = processString(modelName)
          // テキスト色を白色に強制設定
          self.labelName.textColor = .white
          
          // Update new UI
          self.updateUIAfterModelLoad(success: true, modelName: modelName)
        }

        self.downloadProgressLabel.text = "Finished loading model \(modelName)"
        self.downloadProgressLabel.isHidden = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
          self.downloadProgressLabel.isHidden = true
          self.downloadProgressLabel.text = ""
        }

      } else {
        print("Failed to load model: \(modelName)")
      }
    }
  }

  private func convertTaskNameToYOLOTask(_ task: String) -> YOLOTask {
    switch task {
    case "Detect": return .detect
    case "Segment": return .segment
    case "Classify": return .classify
    case "Pose": return .pose
    case "OBB": return .obb
    default: return .detect
    }
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

  @IBAction func indexChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()

    let index = sender.selectedSegmentIndex
    guard tasks.indices.contains(index) else { return }

    let newTask = tasks[index].name

    if (modelsForTask[newTask]?.isEmpty ?? true) && (remoteModelsInfo[newTask]?.isEmpty ?? true) {
      let alert = UIAlertController(
        title: "\(newTask) Models not found",
        message: "Please add or define models for \(newTask).",
        preferredStyle: .alert
      )
      alert.addAction(
        UIAlertAction(
          title: "OK", style: .cancel,
          handler: { _ in
            alert.dismiss(animated: true)
          }))
      self.present(alert, animated: true)

      if let oldIndex = tasks.firstIndex(where: { $0.name == currentTask }) {
        sender.selectedSegmentIndex = oldIndex
      }
      return
    }

    currentTask = newTask
    selectedIndexPath = nil

    reloadModelEntriesAndLoadFirst(for: currentTask)

    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: CGFloat(currentModels.count * 30 + 2)
    )
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
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

    tableViewBGView.backgroundColor = .darkGray.withAlphaComponent(0.3)
    tableViewBGView.layer.cornerRadius = 5  // 選択時の枠のcorner radiusに合わせる
    tableViewBGView.clipsToBounds = true

    view.addSubview(tableViewBGView)
    view.addSubview(modelTableView)

    modelTableView.translatesAutoresizingMaskIntoConstraints = false
    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: CGFloat(currentModels.count * 30 + 2)
    )
  }

  private func setupButtons() {
    // Only setup old UI buttons if not using new UI
    if !isNewUIActive {
      let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)
      shareButton.setImage(
        UIImage(systemName: "square.and.arrow.up", withConfiguration: config), for: .normal)
      shareButton.addGestureRecognizer(
        UITapGestureRecognizer(target: self, action: #selector(shareButtonTapped)))
      view.addSubview(shareButton)

      recordButton.setImage(UIImage(systemName: "video", withConfiguration: config), for: .normal)
      recordButton.addGestureRecognizer(
        UITapGestureRecognizer(target: self, action: #selector(recordScreen)))
      view.addSubview(recordButton)

      logoImage.isUserInteractionEnabled = true
      logoImage.addGestureRecognizer(
        UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    }
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()

    // Only layout old UI elements if not using new UI
    if !isNewUIActive {
      if view.bounds.width > view.bounds.height {
        shareButton.tintColor = .darkGray
        recordButton.tintColor = .darkGray
        let tableViewWidth = view.bounds.width * 0.2
        modelTableView.frame = CGRect(
          x: segmentedControl.frame.maxX + 20, y: 20, width: tableViewWidth, height: 200)
      } else {
        shareButton.tintColor = .systemGray
        recordButton.tintColor = .systemGray
        let tableViewWidth = view.bounds.width * 0.4
        modelTableView.frame = CGRect(
          x: view.bounds.width - tableViewWidth - 8,
          y: segmentedControl.frame.maxY + 25,
          width: tableViewWidth,
          height: 200)
      }

      shareButton.frame = CGRect(
        x: view.bounds.maxX - 49.5,
        y: view.bounds.maxY - 66,
        width: 49.5,
        height: 49.5
      )
      recordButton.frame = CGRect(
        x: shareButton.frame.minX - 49.5,
        y: view.bounds.maxY - 66,
        width: 49.5,
        height: 49.5
      )

      tableViewBGView.frame = CGRect(
        x: modelTableView.frame.minX - 1,
        y: modelTableView.frame.minY - 1,
        width: modelTableView.frame.width + 2,
        height: CGFloat(currentModels.count * 30 + 2)
      )
    }
  }

  @objc func shareButtonTapped() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] captured in
      guard let self = self else { return }
      if let image = captured {
        DispatchQueue.main.async {
          let activityViewController = UIActivityViewController(
            activityItems: [image], applicationActivities: nil
          )
          activityViewController.popoverPresentationController?.sourceView = self.View0
          self.present(activityViewController, animated: true, completion: nil)
        }
      } else {
        print("error capturing photo")
      }
    }
  }

  @objc func recordScreen() {
    let recorder = RPScreenRecorder.shared()
    recorder.isMicrophoneEnabled = true

    if !recorder.isRecording {
      AudioServicesPlaySystemSound(1117)
      recordButton.tintColor = .red
      recorder.startRecording { error in
        if let error = error {
          print("Screen recording start error: \(error)")
        } else {
          print("Started screen recording.")
        }
      }
    } else {
      AudioServicesPlaySystemSound(1118)
      if view.bounds.width > view.bounds.height {
        recordButton.tintColor = .darkGray
      } else {
        recordButton.tintColor = .systemGray
      }
      recorder.stopRecording { previewVC, error in
        if let error = error {
          print("Stop recording error: \(error)")
        }
        if let previewVC = previewVC {
          previewVC.previewControllerDelegate = self
          self.present(previewVC, animated: true, completion: nil)
        }
      }
    }
  }
}

// MARK: - UITableViewDataSource, UITableViewDelegate
extension ViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return currentModels.count
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    return 30
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    // Get custom cell
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

  // This method is no longer needed as we adjust cell background layout in layoutSubviews

}

extension ViewController: RPPreviewViewControllerDelegate {
  func previewControllerDidFinish(_ previewController: RPPreviewViewController) {
    previewController.dismiss(animated: true)
  }
}

// MARK: - YOLOViewDelegate
extension ViewController {
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
    labelFPS.textColor = .white
    
    // Update new UI metrics
    if isNewUIActive {
      statusMetricBar.updateMetrics(fps: fps, latency: inferenceTime)
    }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    DispatchQueue.main.async {
    }
  }

}

// MARK: - New UI Setup
extension ViewController {
  private func setupNewUI() {
    view.backgroundColor = .ultralyticsSurfaceDark
    
    // Camera Preview Container
    cameraPreviewContainer.backgroundColor = .black
    cameraPreviewContainer.layer.cornerRadius = 18
    cameraPreviewContainer.clipsToBounds = true
    
    // Add components to view (order matters for z-index)
    [statusMetricBar, cameraPreviewContainer, taskTabStrip, shutterBar, rightSideToolBar, parameterEditView].forEach {
      view.addSubview($0)
      $0.translatesAutoresizingMaskIntoConstraints = false
    }
    
    // Add model dropdown last so it appears on top
    view.addSubview(modelDropdown)
    modelDropdown.translatesAutoresizingMaskIntoConstraints = false
    
    // Move YOLOView to camera preview container
    if let yoloView = yoloView {
      yoloView.removeFromSuperview()
      cameraPreviewContainer.addSubview(yoloView)
      yoloView.translatesAutoresizingMaskIntoConstraints = false
      NSLayoutConstraint.activate([
        yoloView.topAnchor.constraint(equalTo: cameraPreviewContainer.topAnchor),
        yoloView.leadingAnchor.constraint(equalTo: cameraPreviewContainer.leadingAnchor),
        yoloView.trailingAnchor.constraint(equalTo: cameraPreviewContainer.trailingAnchor),
        yoloView.bottomAnchor.constraint(equalTo: cameraPreviewContainer.bottomAnchor)
      ])
      
      // Ensure user interaction is enabled
      cameraPreviewContainer.isUserInteractionEnabled = true
      yoloView.isUserInteractionEnabled = true
    }
    
    setupNewUIConstraints()
    setupNewUIActions()
    
    // Load models for the new UI
    loadModelsForAllTasks()
    
    // Initial task setup - this will trigger model loading
    taskTabStrip.selectedTask = .detect
    handleTaskChange(to: .detect)
    
    // Listen for hidden info notification
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(showHiddenInfo),
      name: .showHiddenInfo,
      object: nil
    )
  }
  
  private func setupNewUIConstraints() {
    NSLayoutConstraint.activate([
      // Status Bar
      statusMetricBar.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
      statusMetricBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      statusMetricBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      
      // Shutter Bar (set this first as reference)
      shutterBar.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
      shutterBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      shutterBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      
      // Task Tab Strip (position above shutter bar)
      taskTabStrip.bottomAnchor.constraint(equalTo: shutterBar.topAnchor),
      taskTabStrip.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      taskTabStrip.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      
      // Camera Preview (fill space between status bar and task tab strip)
      cameraPreviewContainer.topAnchor.constraint(equalTo: statusMetricBar.bottomAnchor, constant: 8),
      cameraPreviewContainer.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 6),
      cameraPreviewContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -6),
      cameraPreviewContainer.bottomAnchor.constraint(equalTo: taskTabStrip.topAnchor, constant: -8),
      
      // Right Tool Bar (positioned near bottom of camera preview)
      rightSideToolBar.trailingAnchor.constraint(equalTo: cameraPreviewContainer.trailingAnchor, constant: -12),
      rightSideToolBar.bottomAnchor.constraint(equalTo: cameraPreviewContainer.bottomAnchor, constant: -20),
      
      // Parameter Edit View (overlay)
      parameterEditView.topAnchor.constraint(equalTo: cameraPreviewContainer.topAnchor),
      parameterEditView.leadingAnchor.constraint(equalTo: cameraPreviewContainer.leadingAnchor),
      parameterEditView.trailingAnchor.constraint(equalTo: cameraPreviewContainer.trailingAnchor),
      parameterEditView.bottomAnchor.constraint(equalTo: taskTabStrip.topAnchor),
      
      // Model Dropdown
      modelDropdown.topAnchor.constraint(equalTo: statusMetricBar.bottomAnchor),
      modelDropdown.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      modelDropdown.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      modelDropdown.bottomAnchor.constraint(equalTo: view.bottomAnchor)
    ])
  }
  
  private func setupNewUIActions() {
    // Status bar actions
    statusMetricBar.onModelTap = { [weak self] in
      self?.showModelSelector()
    }
    
    // Model dropdown delegate
    modelDropdown.delegate = self
    
    // Task tab actions
    taskTabStrip.onTaskChange = { [weak self] task in
      self?.handleTaskChange(to: task)
    }
    
    // Shutter bar actions
    shutterBar.onShutterTap = { [weak self] in
      self?.capturePhoto()
    }
    
    shutterBar.onShutterLongPress = { [weak self] in
      self?.toggleRecording()
    }
    
    shutterBar.onFlipCamera = { [weak self] in
      self?.flipCamera()
    }
    
    shutterBar.onThumbnailTap = { [weak self] in
      self?.showLastCapture()
    }
    
    // Right toolbar actions
    rightSideToolBar.onZoomChanged = { [weak self] zoomLevel in
      self?.handleZoomChange(to: zoomLevel)
    }
    
    rightSideToolBar.onToolSelected = { [weak self] tool in
      self?.handleParameterTool(tool)
    }
    
    // Parameter edit actions
    parameterEditView.onValueChange = { [weak self] parameter in
      self?.handleParameterChange(parameter)
    }
  }
  
  private func hideOldUI() {
    // Hide old UI elements
    segmentedControl.isHidden = true
    labelName.isHidden = true
    labelFPS.isHidden = true
    labelVersion.isHidden = true
    shareButton.isHidden = true
    recordButton.isHidden = true
    modelTableView.isHidden = true
    tableViewBGView.isHidden = true
    forcus.isHidden = true
    logoImage.isHidden = true
    activityIndicator.isHidden = true
    
    // Hide YOLOView's built-in UI elements
    yoloView.toolbar.isHidden = true
    yoloView.sliderConf.isHidden = true
    yoloView.sliderIoU.isHidden = true
    yoloView.sliderNumItems.isHidden = true
    yoloView.labelSliderConf.isHidden = true
    yoloView.labelSliderIoU.isHidden = true
    yoloView.labelSliderNumItems.isHidden = true
    yoloView.playButton.isHidden = true
    yoloView.pauseButton.isHidden = true
    yoloView.switchCameraButton.isHidden = true
    yoloView.labelZoom.isHidden = true
  }
  
  // MARK: - New UI Actions
  
  private func showModelSelector() {
    print("showModelSelector called")
    print("Current models count: \(currentModels.count)")
    print("Current model name: \(currentModelName)")
    
    // Ensure dropdown is on top
    view.bringSubviewToFront(modelDropdown)
    
    // Find the current model's identifier
    let currentModelIdentifier = currentModels.first { model in
      model.displayName == currentModelName
    }?.identifier
    
    print("Current model identifier: \(currentModelIdentifier ?? "nil")")
    
    // Configure and toggle dropdown
    modelDropdown.configure(with: currentModels, currentModel: currentModelIdentifier)
    modelDropdown.toggle()
  }
  
  private func handleTaskChange(to task: TaskTabStrip.Task) {
    // Hide dropdown if it's showing
    if modelDropdown.isShowing {
      modelDropdown.hide()
    }
    
    let taskName: String
    switch task {
    case .detect:
      taskName = "Detect"
    case .segment:
      taskName = "Segment"
    case .classify:
      taskName = "Classify"
    case .pose:
      taskName = "Pose"
    case .obb:
      taskName = "OBB"
    }
    
    currentTask = taskName
    reloadModelEntriesAndLoadFirst(for: taskName)
  }
  
  private func capturePhoto() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] captured in
      guard let self = self else { return }
      if let image = captured {
        // Update thumbnail
        self.shutterBar.updateThumbnail(image)
        
        // Share functionality
        DispatchQueue.main.async {
          let activityViewController = UIActivityViewController(
            activityItems: [image], applicationActivities: nil
          )
          activityViewController.popoverPresentationController?.sourceView = self.shutterBar
          self.present(activityViewController, animated: true, completion: nil)
        }
      } else {
        print("error capturing photo")
      }
    }
  }
  
  private func toggleRecording() {
    let recorder = RPScreenRecorder.shared()
    recorder.isMicrophoneEnabled = true
    
    if !recorder.isRecording {
      AudioServicesPlaySystemSound(1117)
      shutterBar.setRecording(true)
      recorder.startRecording { error in
        if let error = error {
          print("Screen recording start error: \(error)")
        } else {
          print("Started screen recording.")
        }
      }
    } else {
      AudioServicesPlaySystemSound(1118)
      shutterBar.setRecording(false)
      recorder.stopRecording { previewVC, error in
        if let error = error {
          print("Stop recording error: \(error)")
        }
        if let previewVC = previewVC {
          previewVC.previewControllerDelegate = self
          self.present(previewVC, animated: true, completion: nil)
        }
      }
    }
  }
  
  private func flipCamera() {
    // Use YOLOView's switch camera button tap action
    yoloView.switchCameraButton.sendActions(for: .touchUpInside)
  }
  
  private func showLastCapture() {
    // Placeholder for showing last capture
    print("Show last capture")
  }
  
  private func handleZoomChange(to zoomLevel: Float) {
    // Apply zoom using the new public method
    yoloView?.setZoomLevel(zoomLevel)
    
    // Get the actual applied zoom level (may be clamped to device limits)
    if let actualZoom = yoloView?.getZoomLevel() {
      rightSideToolBar.updateZoomLevel(actualZoom)
    }
  }
  
  private func handleParameterTool(_ tool: RightSideToolBar.Tool) {
    switch tool {
    case .itemsMax:
      let current = Int(currentThresholds["itemsMax"] ?? 15)
      parameterEditView.showParameter(.itemsMax(current))
    case .confidence:
      let current = currentThresholds["confidence"] ?? 0.25
      parameterEditView.showParameter(.confidence(current))
    case .iou:
      let current = currentThresholds["iou"] ?? 0.45
      parameterEditView.showParameter(.iou(current))
    case .lineThickness:
      let current = currentThresholds["lineThickness"] ?? 2.0
      parameterEditView.showParameter(.lineThickness(current))
    default:
      break
    }
  }
  
  private func handleParameterChange(_ parameter: ParameterEditView.Parameter) {
    switch parameter {
    case .itemsMax(let value):
      currentThresholds["itemsMax"] = Float(value)
      // Update via slider
      yoloView.sliderNumItems.value = Float(value)
      yoloView.sliderNumItems.sendActions(for: .valueChanged)
    case .confidence(let value):
      currentThresholds["confidence"] = value
      // Update via slider
      yoloView.sliderConf.value = value
      yoloView.sliderConf.sendActions(for: .valueChanged)
    case .iou(let value):
      currentThresholds["iou"] = value
      // Update via slider
      yoloView.sliderIoU.value = value
      yoloView.sliderIoU.sendActions(for: .valueChanged)
    case .lineThickness(let value):
      currentThresholds["lineThickness"] = value
      // Line thickness is not configurable in YOLOView
      print("Line thickness set to: \(value) (not yet implemented)")
    }
    
    // Save to UserDefaults
    UserDefaults.standard.set(currentThresholds, forKey: "thresholds")
  }
  
  @objc private func showHiddenInfo() {
    let hiddenInfoVC = HiddenInfoViewController()
    hiddenInfoVC.modalPresentationStyle = .pageSheet
    present(hiddenInfoVC, animated: true)
  }
  
  private func updateUIAfterModelLoad(success: Bool, modelName: String) {
    
    if success && isNewUIActive {
      // Update status bar with model info
      let modelSize = ModelSizeHelper.getModelSize(from: modelName)
      statusMetricBar.updateModel(name: processString(modelName), size: modelSize)
    }
  }
}

// MARK: - ModelDropdownViewDelegate
extension ViewController {
  func modelDropdown(_ dropdown: ModelDropdownView, didSelectModel model: ModelEntry) {
    loadModel(entry: model, forTask: currentTask)
  }
  
  func modelDropdownDidDismiss(_ dropdown: ModelDropdownView) {
    // Handle dropdown dismissal if needed
  }
}
