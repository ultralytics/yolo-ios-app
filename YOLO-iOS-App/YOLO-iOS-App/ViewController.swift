//
//  ViewController.swift
//  YOLOTest
//
//  Created by Example on 2025/01/25.
//

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import YOLO

class ViewController: UIViewController {

    @IBOutlet weak var yoloView: YOLOView!
    @IBOutlet var View0: UIView!
    @IBOutlet var segmentedControl: UISegmentedControl!
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var focus: UIImageView!
    
    var shareButton = UIButton()
    let selection = UISelectionFeedbackGenerator()
    
    private let tasks: [(name: String, folder: String)] = [
        ("Detect",   "DetectModels"),
        ("Segment",  "SegmentModels"),
        ("Classify", "ClassifyModels"),
        ("Pose",     "PoseModels"),
        ("Obb",      "ObbModels"),
    ]
    
    // タスク名 -> [モデルファイル名] の対応を保持
    private var modelsForTask: [String: [String]] = [:]
    
    // 現在のタスク
    private var currentTask: String = ""
    // 現在読み込んでいるモデル名（ファイル名）
    private var currentModelName: String = ""
    
    // 現在のタスクに対応するモデル一覧を保持する
    private var currentModels: [String] = []
    private var isLoadingModel = false

    // 右半分に表示する TableView
    private let modelTableView: UITableView = {
        let table = UITableView()
        table.isHidden = true // デフォルトは非表示にしておく
        // TableView 自体を角丸に
        table.layer.cornerRadius = 8
        table.clipsToBounds = true
        return table
    }()
    
    private let tableViewBGView = UIView()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupTaskSegmentedControl()
        loadModelsForAllTasks()
        
        // 最初のタスク＆モデルを自動選択状態に
        if tasks.indices.contains(0) {
            segmentedControl.selectedSegmentIndex = 0
            currentTask = tasks[0].name
            loadFirstModel(for: currentTask)
            currentModels = modelsForTask[currentTask] ?? []
            
            if !currentModels.isEmpty {
                modelTableView.isHidden = false
                modelTableView.reloadData()
                
                // reloadData 直後に選択状態を反映
                DispatchQueue.main.async {
                    let firstIndex = IndexPath(row: 0, section: 0)
                    self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
                }
            }
        }
        
        setupTableView()
        setupShareButton()
    }
    
    // セグメントコントロールを初期化
    private func setupTaskSegmentedControl() {
        segmentedControl.removeAllSegments()
        for (index, taskInfo) in tasks.enumerated() {
            segmentedControl.insertSegment(withTitle: taskInfo.name, at: index, animated: false)
        }
    }
    
    // 各タスクフォルダからモデルファイル名一覧を取得して保持
    private func loadModelsForAllTasks() {
        for taskInfo in tasks {
            let taskName = taskInfo.name
            let folderName = taskInfo.folder
            let modelFiles = getModelFiles(in: folderName)
            modelsForTask[taskName] = modelFiles
        }
    }
    
    // 指定フォルダの中にある .mlmodel と .mlpackage のファイル名を取得
    private func getModelFiles(in folderName: String) -> [String] {
        var result: [String] = []
        
        if let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil) {
            do {
                let fileURLs = try FileManager.default.contentsOfDirectory(
                    at: folderURL,
                    includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles]
                )
                for fileURL in fileURLs {
                    if fileURL.pathExtension == "mlmodel" || fileURL.pathExtension == "mlpackage" {
                        let fileName = fileURL.lastPathComponent
                        result.append(fileName)
                    }
                }
            } catch {
                print("Error reading contents of folder \(folderName): \(error)")
            }
        }
        
        return result.sorted()
    }
    
    // 指定タスクの最初のモデルを読み込む
    private func loadFirstModel(for task: String) {
        guard let models = modelsForTask[task], !models.isEmpty else {
            print("No models found for task: \(task)")
            return
        }
        let firstModel = models[0]
        loadModel(named: firstModel, forTask: task)
        
        // TableView 上でも最初の行を選択状態にする（画面表示後に選択が反映されるようにするため、あとでDispatchQueue.main.asyncでも再度実行）
        let firstIndex = IndexPath(row: 0, section: 0)
        modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
    }
    
    private func loadModel(named modelName: String, forTask task: String) {
        guard !isLoadingModel else {
            print("Model is already loading. Please wait.")
            return
        }
        isLoadingModel = true
        
        activityIndicator.startAnimating()
        self.view.isUserInteractionEnabled = false
        modelTableView.isUserInteractionEnabled = false
        
        print("Start loading model: \(modelName)")
        
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            let yoloTask = self.convertTaskNameToYOLOTask(task)

            guard let modelFolder = self.tasks.first(where: { $0.name == task })?.folder,
                  let folderURL = Bundle.main.url(forResource: modelFolder, withExtension: nil) else {
                DispatchQueue.main.async {
                    self.finishLoadingModel(success: false, modelName: modelName)
                }
                return
            }
            
            let modelURL = folderURL.appendingPathComponent(modelName)
            
            DispatchQueue.main.async {
                self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
                    switch result {
                    case .success():
                        self.finishLoadingModel(success: true, modelName: modelName)
                    case .failure(let error):
                        print(error)
                        self.finishLoadingModel(success: false, modelName: modelName)
                    }
                }
            }
        }
    }
    
    private func finishLoadingModel(success: Bool, modelName: String) {
        self.activityIndicator.stopAnimating()
        self.view.isUserInteractionEnabled = true
        self.modelTableView.isUserInteractionEnabled = true
        self.isLoadingModel = false
        
        if success {
            print("Finished loading model: \(modelName)")
            self.currentModelName = modelName
        } else {
            print("Failed to load model: \(modelName)")
        }
    }
    
    private func convertTaskNameToYOLOTask(_ task: String) -> YOLOTask {
        switch task {
        case "Detect":   return .detect
        case "Segment":  return .segment
        case "Classify": return .classify
        case "Pose":     return .pose
        case "Obb":      return .obb
        default:         return .detect
        }
    }
    
    @IBAction func vibrate(_ sender: Any) {
        selection.selectionChanged()
    }
    
    @IBAction func indexChanged(_ sender: UISegmentedControl) {
        // セグメント選択時にもフィードバック
        selection.selectionChanged()
        
        let index = sender.selectedSegmentIndex
        guard tasks.indices.contains(index) else { return }
        
        let newTask = tasks[index].name
        currentTask = newTask
        currentModels = modelsForTask[currentTask] ?? []
        
        if !currentModels.isEmpty {
            // タスク変更時も最初のモデルをロード
            loadModel(named: currentModels[0], forTask: currentTask)
            modelTableView.isHidden = false
            modelTableView.reloadData()
            
            DispatchQueue.main.async {
                let firstIndex = IndexPath(row: 0, section: 0)
                self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
            }
        } else {
            print("No models available for task: \(currentTask)")
            modelTableView.isHidden = true
        }
        tableViewBGView.frame = CGRect(x: modelTableView.frame.minX-1, y: modelTableView.frame.minY-1, width: modelTableView.frame.width+2, height: CGFloat(modelsForTask[currentTask]!.count*30+2))
        print(tableViewBGView.frame)
    }
    
    @IBAction func logoButton(_ sender: Any) {
        selection.selectionChanged()
        if let link = URL(string: "https://www.ultralytics.com") {
            UIApplication.shared.open(link)
        }
    }
    
    private func setupTableView() {
        modelTableView.delegate = self
        modelTableView.dataSource = self
        modelTableView.register(UITableViewCell.self, forCellReuseIdentifier: "ModelCell")
        modelTableView.backgroundColor = .clear
        // セル同士の区切り線を非表示
        modelTableView.separatorStyle = .none
        tableViewBGView.backgroundColor = .darkGray.withAlphaComponent(0.3)
        tableViewBGView.layer.cornerRadius = 8
        tableViewBGView.clipsToBounds = true
        view.addSubview(tableViewBGView)

        view.addSubview(modelTableView)
        modelTableView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            modelTableView.topAnchor.constraint(equalTo: segmentedControl.bottomAnchor, constant: 25),
            // TableView の右端を segmentedControl の右端に揃える
            modelTableView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -8),
            // 左端は「幅を全体の40%」→ 元のコードと同様
            modelTableView.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.4),
            modelTableView.heightAnchor.constraint(equalToConstant: 200)
        ])
        tableViewBGView.frame = CGRect(x: modelTableView.frame.minX-1, y: modelTableView.frame.minY-1, width: modelTableView.frame.width+2, height: CGFloat(modelsForTask[currentTask]!.count*30+2))

    }
    
    private func setupShareButton() {
        let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)
        shareButton.setImage(UIImage(systemName: "square.and.arrow.up", withConfiguration: config), for: .normal)
        shareButton.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(shareButtonTapped)))
        view.addSubview(shareButton)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        if view.bounds.width > view.bounds.height {
            shareButton.tintColor = .darkGray
        } else {
            shareButton.tintColor = .systemGray
        }
        
        shareButton.frame = CGRect(
            x: view.bounds.maxX - 49.5,
            y: view.bounds.maxY - 66,
            width: 49.5,
            height: 49.5
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
    
    // セルの数
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return currentModels.count
    }
    
    // セルの高さを固定して（文字サイズに合わせてやや小さめに）
    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 30
    }
    
    // セルの中身
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {

        let cell = tableView.dequeueReusableCell(withIdentifier: "ModelCell", for: indexPath)
        
        // 拡張子を削除して表示
        let fileName = currentModels[indexPath.row]
        let displayName = (fileName as NSString).deletingPathExtension
        
        // テキストを Heavy に & 中央揃え
        cell.textLabel?.textAlignment = .center
        cell.textLabel?.text = displayName
        cell.textLabel?.font = UIFont.systemFont(ofSize: 14, weight: .medium)
        
        // TableView全体を角丸にしたので、セルの背景は透明に
        cell.backgroundColor = .clear
        
        // 選択時のバックグラウンドビューを作成（セルより少し小さく + 角丸）
        let selectedBGView = UIView()
        selectedBGView.backgroundColor = UIColor(white: 1.0, alpha: 0.3)
        selectedBGView.layer.cornerRadius = 8
        selectedBGView.layer.masksToBounds = true
        selectedBGView.frame = CGRect(x: 2, y:2, width: cell.contentView.bounds.width - 4, height: cell.contentView.bounds.height - 4)
        cell.selectedBackgroundView = selectedBGView
        
        // selectionStyle は .default のままにして、選択演出あり
        cell.selectionStyle = .default
        
        return cell
    }
    
    // セルをタップしたときの処理
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let selectedModel = currentModels[indexPath.row]
        
        // セル選択時にもフィードバック
        selection.selectionChanged()
        
        // モデルをロード
        loadModel(named: selectedModel, forTask: currentTask)
    }
    
    // セルが描画される直前に呼ばれる
    // 選択背景をセル本体より少し小さく見せるために frame を調整
    func tableView(_ tableView: UITableView,
                   willDisplay cell: UITableViewCell,
                   forRowAt indexPath: IndexPath) {
        if let selectedBGView = cell.selectedBackgroundView {
            // 例えば左右/上下に4ptずつ内側にオフセット
            let insetRect = cell.bounds.insetBy(dx: 4, dy: 4)
            selectedBGView.frame = insetRect
        }
    }
}
