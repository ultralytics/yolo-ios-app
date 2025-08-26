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
  var isSuccess: Bool {
    if case .success = self { return true }
    return false
  }
}

// MARK: - ModelTableViewCell
class ModelTableViewCell: UITableViewCell {
  static let identifier = "ModelTableViewCell"

  override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
    super.init(style: .default, reuseIdentifier: reuseIdentifier)
    setupCell()
  }

  required init?(coder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }

  private func setupCell() {
    backgroundColor = .clear
    selectionStyle = .default
    textLabel?.textAlignment = .center
    textLabel?.font = .systemFont(ofSize: 14, weight: .medium)
    textLabel?.adjustsFontSizeToFitWidth = true
    textLabel?.minimumScaleFactor = 0.7
    textLabel?.textColor = .white

    let selectedBGView = UIView()
    selectedBGView.backgroundColor = UIColor(white: 1.0, alpha: 0.3)
    selectedBGView.layer.cornerRadius = 5
    selectedBackgroundView = selectedBGView
  }

  func configure(with modelName: String, isRemote: Bool, isDownloaded: Bool) {
    textLabel?.text = modelName
    textLabel?.textColor = (isRemote && !isDownloaded) ? .lightGray : .white

    if isRemote && !isDownloaded {
      let config = UIImage.SymbolConfiguration(pointSize: 14)
      imageView?.image = UIImage(systemName: "icloud.and.arrow.down", withConfiguration: config)
      imageView?.tintColor = .white
    } else {
      imageView?.image = nil
    }
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    selectedBackgroundView?.frame = bounds.insetBy(dx: 2, dy: 1)
  }
}

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewDelegate {

  @IBOutlet weak var yoloView: YOLOView!
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var logoImage: UIImageView!

  let selection = UISelectionFeedbackGenerator()

  private let downloadProgressView = UIProgressView(progressViewStyle: .default)
  private let downloadProgressLabel = UILabel()

  private var loadingOverlayView: UIView?

  // MARK: - Loading State Management
  private func setLoadingState(_ loading: Bool, showOverlay: Bool = false) {
    if loading {
      activityIndicator.startAnimating()
      view.isUserInteractionEnabled = false
      modelTableView.isUserInteractionEnabled = false

      if showOverlay {
        showLoadingOverlay()
      }
    } else {
      activityIndicator.stopAnimating()
      view.isUserInteractionEnabled = true
      modelTableView.isUserInteractionEnabled = true
      hideLoadingOverlay()
    }
  }

  private func showLoadingOverlay() {
    guard loadingOverlayView == nil else { return }
    let overlay = UIView(frame: view.bounds)
    overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)
    view.addSubview(overlay)
    loadingOverlayView = overlay
    view.bringSubviewToFront(downloadProgressView)
    view.bringSubviewToFront(downloadProgressLabel)
  }

  private func hideLoadingOverlay() {
    loadingOverlayView?.removeFromSuperview()
    loadingOverlayView = nil
  }

  private let tasks: [(name: String, folder: String)] = [
    ("Classify", "ClassifyModels"),  // index 0
    ("Segment", "SegmentModels"),  // index 1
    ("Detect", "DetectModels"),  // index 2
    ("Pose", "PoseModels"),  // index 3
    ("OBB", "OBBModels"),  // index 4
  ]

  private var modelsForTask: [String: [String]] = [:]

  private var currentModels: [(name: String, url: URL?, isLocal: Bool)] = []

  private var currentTask: String = ""
  private var currentModelName: String = ""

  private var isLoadingModel = false

  // MARK: - Constants
  private struct Constants {
    static let defaultTaskIndex = 2  // Detect
    static let tableRowHeight: CGFloat = 30
    static let logoURL = "https://www.ultralytics.com"
    static let progressViewWidth: CGFloat = 200
    static let activityIndicatorSize: CGFloat = 100
  }

  private let modelTableView: UITableView = {
    let table = UITableView()
    table.isHidden = true
    table.layer.cornerRadius = 5  // Match corner radius of other elements
    table.clipsToBounds = true
    return table
  }()

  private let tableViewBGView = UIView()

  private var selectedIndexPath: IndexPath?

  override func viewDidLoad() {
    super.viewDidLoad()

    setupTaskSegmentedControl()
    loadModelsForAllTasks()

    if tasks.indices.contains(Constants.defaultTaskIndex) {
      segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
      currentTask = tasks[Constants.defaultTaskIndex].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    setupTableView()

    // Setup logo tap gesture
    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))

    // Setup share button
    yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)

    yoloView.delegate = self
    yoloView.labelName.isHidden = true
    yoloView.labelFPS.isHidden = true

    // Setup labels once
    setupLabels()

    setupDownloadProgressViews()
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    view.overrideUserInterfaceStyle = .dark
  }

  // MARK: - Setup Methods
  private func setupDownloadProgressViews() {
    downloadProgressView.isHidden = true
    downloadProgressLabel.textAlignment = .center
    downloadProgressLabel.textColor = .systemGray
    downloadProgressLabel.font = .systemFont(ofSize: 14)
    downloadProgressLabel.isHidden = true

    [downloadProgressView, downloadProgressLabel].forEach {
      $0.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview($0)
    }

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
  }

  private func setupLabels() {
    labelName.textColor = .white
    labelFPS.textColor = .white
    labelVersion.textColor = .white

    labelName.overrideUserInterfaceStyle = .dark
    labelFPS.overrideUserInterfaceStyle = .dark
    labelVersion.overrideUserInterfaceStyle = .dark

    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
      let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    {
      labelVersion.text = "v\(version) (\(build))"
    }
  }

  private func setupTaskSegmentedControl() {
    segmentedControl.removeAllSegments()
    for (index, taskInfo) in tasks.enumerated() {
      segmentedControl.insertSegment(withTitle: taskInfo.name, at: index, animated: false)
    }
  }

  private func loadModelsForAllTasks() {
    tasks.forEach { taskInfo in
      modelsForTask[taskInfo.name] = getModelFiles(in: taskInfo.folder)
    }
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
    let officialOrder: [Character: Int] = ["n": 0, "m": 1, "s": 2, "l": 3, "x": 4]

    let (official, custom) = fileNames.reduce(into: ([String](), [String]())) { result, fileName in
      let baseName = (fileName as NSString).deletingPathExtension.lowercased()
      if baseName.hasPrefix("yolo"), let lastChar = baseName.last, officialOrder[lastChar] != nil {
        result.0.append(fileName)
      } else {
        result.1.append(fileName)
      }
    }

    return custom.sorted()
      + official.sorted { fileA, fileB in
        let baseA = (fileA as NSString).deletingPathExtension.lowercased()
        let baseB = (fileB as NSString).deletingPathExtension.lowercased()
        let indexA = baseA.last.flatMap { officialOrder[$0] } ?? Int.max
        let indexB = baseB.last.flatMap { officialOrder[$0] } ?? Int.max
        return indexA < indexB
      }
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

  private func makeModelEntries(for taskName: String) -> [(name: String, url: URL?, isLocal: Bool)]
  {
    let localFileNames = modelsForTask[taskName] ?? []
    let localModels = localFileNames.map { fileName in
      let displayName = (fileName as NSString).deletingPathExtension
      return (name: displayName, url: nil as URL?, isLocal: true)
    }

    // Get local model names for filtering
    let localModelNames = Set(localModels.map { $0.name.lowercased() })

    let remoteList = remoteModelsInfo[taskName] ?? []
    let remoteModels = remoteList.compactMap {
      (modelName, url) -> (name: String, url: URL?, isLocal: Bool)? in
      // Only include remote models if no local model with the same name exists
      guard !localModelNames.contains(modelName.lowercased()) else { return nil }
      return (name: modelName, url: url, isLocal: false)
    }

    return localModels + remoteModels
  }

  private func loadModel(entry: (name: String, url: URL?, isLocal: Bool), forTask task: String) {
    guard !isLoadingModel else { return }
    isLoadingModel = true

    yoloView.resetLayers()
    yoloView.setInferenceFlag(ok: false)
    setLoadingState(true, showOverlay: true)
    resetDownloadProgress()

    let yoloTask = convertTaskNameToYOLOTask(task)

    if entry.isLocal {
      loadLocalModel(entry: entry, task: task, yoloTask: yoloTask)
    } else if let remoteURL = entry.url {
      loadRemoteModel(url: remoteURL, entry: entry, yoloTask: yoloTask)
    } else {
      finishLoadingModel(success: false, modelName: entry.name)
    }
  }

  private func loadLocalModel(
    entry: (name: String, url: URL?, isLocal: Bool), task: String, yoloTask: YOLOTask
  ) {
    guard let folderURL = tasks.first(where: { $0.name == task })?.folder,
      let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
    else {
      finishLoadingModel(success: false, modelName: entry.name)
      return
    }

    let modelURL = folderPathURL.appendingPathComponent(entry.name + ".mlpackage")
    downloadProgressLabel.text = "Loading \(entry.name)"
    downloadProgressLabel.isHidden = false

    yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { [weak self] result in
      self?.finishLoadingModel(success: result.isSuccess, modelName: entry.name)
    }
  }

  private func loadRemoteModel(
    url: URL, entry: (name: String, url: URL?, isLocal: Bool), yoloTask: YOLOTask
  ) {
    let downloader = YOLOModelDownloader()
    downloadProgressView.isHidden = false
    downloadProgressLabel.isHidden = false

    downloader.download(
      from: url, task: yoloTask,
      progress: { [weak self] progress in
        DispatchQueue.main.async {
          self?.downloadProgressView.progress = Float(progress)
          self?.downloadProgressLabel.text = "Downloading \(Int(progress * 100))%"
        }
      },
      completion: { [weak self] result in
        guard let self = self else { return }
        switch result {
        case .success(let modelPath):
          DispatchQueue.main.async {
            self.downloadProgressLabel.text = "Loading \(entry.name)"
            self.yoloView.setModel(modelPathOrName: modelPath.path, task: yoloTask) { result in
              self.finishLoadingModel(success: result.isSuccess, modelName: entry.name)
            }
          }
        case .failure:
          self.finishLoadingModel(success: false, modelName: entry.name)
        }
      }
    )
  }

  private func resetDownloadProgress() {
    downloadProgressView.progress = 0.0
    downloadProgressView.isHidden = true
    downloadProgressLabel.isHidden = true
  }

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async {
      self.setLoadingState(false)
      self.isLoadingModel = false
      self.resetDownloadProgress()

      self.modelTableView.reloadData()
      if let ip = self.selectedIndexPath {
        self.modelTableView.selectRow(at: ip, animated: false, scrollPosition: .none)
      }

      self.yoloView.setInferenceFlag(ok: success)

      if success {
        self.currentModelName = modelName
        self.labelName.text = processString(modelName)
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

    updateTableViewBackground()
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

    tableViewBGView.backgroundColor = .darkGray.withAlphaComponent(0.3)
    tableViewBGView.layer.cornerRadius = 5  // 選択時の枠のcorner radiusに合わせる
    tableViewBGView.clipsToBounds = true

    yoloView.addSubview(tableViewBGView)
    yoloView.addSubview(modelTableView)

    modelTableView.translatesAutoresizingMaskIntoConstraints = false
    updateTableViewBackground()
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    layoutModelTableView()
  }

  private func layoutModelTableView() {
    let isLandscape = view.bounds.width > view.bounds.height
    let tableViewWidth = view.bounds.width * (isLandscape ? 0.2 : 0.4)

    if isLandscape {
      modelTableView.frame = CGRect(
        x: segmentedControl.frame.maxX + 20, y: 20, width: tableViewWidth, height: 200)
    } else {
      modelTableView.frame = CGRect(
        x: view.bounds.width - tableViewWidth - 8,
        y: segmentedControl.frame.maxY + 25,
        width: tableViewWidth,
        height: 200)
    }

    updateTableViewBackground()
  }

  private func updateTableViewBackground() {
    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: CGFloat(currentModels.count * Int(Constants.tableRowHeight) + 2)
    )
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

}

// MARK: - UITableViewDataSource, UITableViewDelegate
extension ViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return currentModels.count
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    return Constants.tableRowHeight
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    // Get custom cell
    let cell =
      tableView.dequeueReusableCell(withIdentifier: ModelTableViewCell.identifier, for: indexPath)
      as! ModelTableViewCell
    let entry = currentModels[indexPath.row]

    // Check if the model is remote and not yet downloaded
    let yoloTask = self.convertTaskNameToYOLOTask(currentTask)
    let isDownloaded =
      entry.isLocal
      || (entry.url != nil && YOLOModelCache.shared.isCached(url: entry.url!, task: yoloTask))

    // Format model name using the processString function
    let formattedName = processString(entry.name)

    // Configure the cell
    cell.configure(with: formattedName, isRemote: !entry.isLocal, isDownloaded: isDownloaded)

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
    labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
    labelFPS.textColor = .white
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    DispatchQueue.main.async {
    }
  }
}
