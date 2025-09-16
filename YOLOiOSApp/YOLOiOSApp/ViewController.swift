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

  @IBOutlet weak var yoloView: YOLOView!, View0: UIView!, segmentedControl: UISegmentedControl!, 
    labelName: UILabel!, labelFPS: UILabel!, labelVersion: UILabel!,
                     activityIndicator: UIActivityIndicatorView!, logoImage: UIImageView!, modelSegmentedControl: UISegmentedControl!

  let selection = UISelectionFeedbackGenerator()

  private let downloadProgressView = UIProgressView(progressViewStyle: .default)
  private let downloadProgressLabel = UILabel()

  private var loadingOverlayView: UIView?

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

  private let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
    ("Classify", "ClassifyModels", .classify),
    ("Segment", "SegmentModels", .segment),
    ("Detect", "DetectModels", .detect),
    ("Pose", "PoseModels", .pose),
    ("OBB", "OBBModels", .obb),
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

    // Setup segmented control and load models
    segmentedControl.removeAllSegments()
    tasks.enumerated().forEach { index, task in
      segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
      modelsForTask[task.name] = getModelFiles(in: task.folder)
    }

    if tasks.indices.contains(Constants.defaultTaskIndex) {
      segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
      currentTask = tasks[Constants.defaultTaskIndex].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    setupTableView()

    modelSegmentedControl.isHidden = false
    modelSegmentedControl.overrideUserInterfaceStyle = .dark
    modelSegmentedControl.apportionsSegmentWidthsByContent = true

    // Setup gestures and delegates
    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    yoloView.delegate = self
    [yoloView.labelName, yoloView.labelFPS].forEach { $0?.isHidden = true }

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

  private func makeModelEntries(for taskName: String) -> [(name: String, url: URL?, isLocal: Bool)]
  {
    let localModels = (modelsForTask[taskName] ?? []).map {
      (name: ($0 as NSString).deletingPathExtension, url: nil as URL?, isLocal: true)
    }
    let localModelNames = Set(localModels.map { $0.name.lowercased() })
    let remoteModels = (remoteModelsInfo[taskName] ?? []).compactMap { modelName, url in
      localModelNames.contains(modelName.lowercased())
        ? nil : (name: modelName, url: url, isLocal: false)
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

    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
    let loadWithPath = { [weak self] (path: String) in
      self?.downloadProgressLabel.text = "Loading \(entry.name)"
      self?.downloadProgressLabel.isHidden = false
      self?.yoloView.setModel(modelPathOrName: path, task: yoloTask) { result in
        self?.finishLoadingModel(success: result.isSuccess, modelName: entry.name)
      }
    }

    if entry.isLocal {
      guard let folderPath = tasks.first(where: { $0.name == task })?.folder,
        let url = Bundle.main.url(forResource: folderPath, withExtension: nil)
      else {
        return finishLoadingModel(success: false, modelName: entry.name)
      }
      loadWithPath(url.appendingPathComponent(entry.name + ".mlpackage").path)
    } else if let remoteURL = entry.url {
      [downloadProgressView, downloadProgressLabel].forEach { $0.isHidden = false }
      YOLOModelDownloader().download(
        from: remoteURL, task: yoloTask,
        progress: { [weak self] progress in
          DispatchQueue.main.async {
            self?.downloadProgressView.progress = Float(progress)
            self?.downloadProgressLabel.text = "Downloading \(Int(progress * 100))%"
          }
        },
        completion: { [weak self] result in
          switch result {
          case .success(let path): DispatchQueue.main.async { loadWithPath(path.path) }
          case .failure: self?.finishLoadingModel(success: false, modelName: entry.name)
          }
        })
    } else {
      finishLoadingModel(success: false, modelName: entry.name)
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
    let yoloTask = tasks.first(where: { $0.name == currentTask })?.yoloTask ?? .detect
    let isDownloaded =
      entry.isLocal
      || (entry.url != nil && YOLOModelCache.shared.isCached(url: entry.url!, task: yoloTask))
    cell.configure(
      with: processString(entry.name), isRemote: !entry.isLocal, isDownloaded: isDownloaded)
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
  }
}
