// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit
import YOLO
import CoreMedia
import AVFoundation

class ExternalViewController: UIViewController, YOLOViewDelegate {
    
    private var yoloView: YOLOView?
    private var isInitialized = false
    private var currentTask: YOLOTask = .detect
    private var currentModelName: String = "yolo11n.mlmodel"
    
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
        ("OBB", .obb)
    ]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        print("🟢 ExternalViewController viewDidLoad")
        
        // Set background color to verify view is visible
        view.backgroundColor = .black
        
        // Configure for full screen display
        edgesForExtendedLayout = .all
        extendedLayoutIncludesOpaqueBars = true
        
        setupUI()
        setupNotifications()
        
        // Debug: Check if we're on the main thread
        print("🟢 Is main thread: \(Thread.isMainThread)")
    }
    
    
    private func setupUI() {
        // Set black background
        view.backgroundColor = .black
        
        print("🟢 Setting up UI for external display")
        print("🟢 View bounds: \(view.bounds)")
        
        setupControlUI()
        
        // Delay YOLOView creation to ensure proper initialization
        // This will be done in viewDidAppear when the view is fully ready
    }
    
    private func setupControlUI() {
        let screenSize = view.bounds.size
        let scaleFactor = calculateScaleFactor(for: screenSize)
        
        print("📐 External display setup:")
        print("  - Screen size: \(screenSize)")
        print("  - Scale factor: \(scaleFactor)")
        print("  - Model name font size: \(screenSize.height * 0.1)pt (2.5% of height)")
        print("  - FPS font size: \(screenSize.height * 0.04)pt (1.67% of height)")
        print("  - Segment control font size: 60pt (fixed)")
        
        // Calculate proportional font sizes based on screen height
        let baseFontSizeModelName = screenSize.height * 0.1 // 2.5% of screen height
        let baseFontSizeFPS = screenSize.height * 0.04 // 1.67% of screen height
        
        // Model name label - reasonable size for external displays
        labelName = UILabel()
        labelName.text = currentModelName
        labelName.textAlignment = .center
        labelName.font = UIFont.systemFont(ofSize: baseFontSizeModelName, weight: .bold) // Proportional to screen
        labelName.textColor = .white
        labelName.adjustsFontSizeToFitWidth = true
        labelName.minimumScaleFactor = 0.5
        labelName.numberOfLines = 1
        labelName.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(labelName)
        
        // FPS label - reasonable size
        labelFPS = UILabel()
        labelFPS.text = "0.0 FPS - 0.0 ms"
        labelFPS.textAlignment = .center
        labelFPS.textColor = .white
        labelFPS.font = UIFont.systemFont(ofSize: baseFontSizeFPS, weight: .medium) // Proportional to screen
        labelFPS.adjustsFontSizeToFitWidth = true
        labelFPS.minimumScaleFactor = 0.5
        labelFPS.numberOfLines = 1
        labelFPS.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(labelFPS)
        
        // Task segmented control - styled like iPhone version
        segmentedControl = UISegmentedControl()
        for (index, taskInfo) in tasks.enumerated() {
            segmentedControl.insertSegment(withTitle: taskInfo.name, at: index, animated: false)
        }
        segmentedControl.selectedSegmentIndex = 2 // Default to Detect
        
        // Style to match iPhone appearance with lighter background
        segmentedControl.backgroundColor = UIColor(white: 0.2, alpha: 0.3)
        segmentedControl.selectedSegmentTintColor = UIColor(white: 0.4, alpha: 0.8)
        segmentedControl.layer.cornerRadius = 12
        segmentedControl.layer.masksToBounds = true
        
        // Text attributes with reasonable font for external display
        let fontSize: CGFloat = 36 // Reasonable size
        segmentedControl.setTitleTextAttributes([
            .foregroundColor: UIColor.white,
            .font: UIFont.systemFont(ofSize: fontSize, weight: .semibold)
        ], for: .selected)
        segmentedControl.setTitleTextAttributes([
            .foregroundColor: UIColor.lightGray,
            .font: UIFont.systemFont(ofSize: fontSize, weight: .medium)
        ], for: .normal)
        
        segmentedControl.isUserInteractionEnabled = false // Controlled by iPhone
        segmentedControl.translatesAutoresizingMaskIntoConstraints = false
        segmentedControl.setContentHuggingPriority(.required, for: .vertical)
        segmentedControl.setContentCompressionResistancePriority(.required, for: .vertical)
        segmentedControl.isHidden = true // Hide task segment control
        view.addSubview(segmentedControl)
        
        // Logo ImageView - bottom right corner, smaller
        logoImageView = UIImageView()
        logoImageView.image = UIImage(named: "ultralytics_yolo_logotype")
        logoImageView.contentMode = .scaleAspectFit
        logoImageView.alpha = 1.0 // Fully opaque
        logoImageView.isUserInteractionEnabled = false // Allow touches to pass through
        logoImageView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(logoImageView)
        
        // Ensure labels have proper content priority
        labelName.setContentHuggingPriority(.required, for: .vertical)
        labelName.setContentCompressionResistancePriority(.required, for: .vertical)
        labelFPS.setContentHuggingPriority(.required, for: .vertical)
        labelFPS.setContentCompressionResistancePriority(.required, for: .vertical)
        
        setupConstraints(scaleFactor: scaleFactor)
    }
    
    private func calculateScaleFactor(for screenSize: CGSize) -> CGFloat {
        let baseSize: CGFloat = 375.0 // iPhone reference size
        let rawScale = max(screenSize.width, screenSize.height) / baseSize
        
        // For text elements, use more aggressive scaling for large displays
        let scaleFactor = 1.0 + (rawScale - 1.0) * 1.2
        
        return max(2.5, min(scaleFactor, 10.0)) // Much higher minimum and maximum for large displays
    }
    
    private func setupConstraints(scaleFactor: CGFloat) {
        let margin: CGFloat = 20
        
        NSLayoutConstraint.activate([
            // Model name label - top center
            labelName.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            labelName.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: margin),
            labelName.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.9),
            labelName.heightAnchor.constraint(equalTo: view.heightAnchor, multiplier: 0.1), // 10% of screen height like iPhone
            
            // FPS label - below model name
            labelFPS.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            labelFPS.topAnchor.constraint(equalTo: labelName.bottomAnchor, constant: 20),
            labelFPS.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.9),
            labelFPS.heightAnchor.constraint(equalTo: view.heightAnchor, multiplier: 0.04), // 4% of screen height like iPhone
            
            // Segmented control - below FPS
            segmentedControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            segmentedControl.topAnchor.constraint(equalTo: labelFPS.bottomAnchor, constant: 30),
            segmentedControl.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.7), // 70% width
            segmentedControl.heightAnchor.constraint(equalToConstant: 60), // Reasonable height for 36pt font
            
            // Logo - bottom right corner, 20% of screen width
            logoImageView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -margin),
            logoImageView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -margin),
            logoImageView.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.2), // 20% of screen width
            logoImageView.heightAnchor.constraint(equalTo: logoImageView.widthAnchor, multiplier: 0.3) // Maintain aspect ratio
        ])
    }
    
    private func hideYOLOViewControls() {
        guard let yoloView = yoloView else { return }
        
        // Hide and remove all sliders from view hierarchy
        yoloView.sliderNumItems.isHidden = true
        yoloView.sliderNumItems.removeFromSuperview()
        yoloView.labelSliderNumItems.isHidden = true
        yoloView.labelSliderNumItems.removeFromSuperview()
        
        yoloView.sliderConf.isHidden = true
        yoloView.sliderConf.removeFromSuperview()
        yoloView.labelSliderConf.isHidden = true
        yoloView.labelSliderConf.removeFromSuperview()
        
        yoloView.sliderIoU.isHidden = true
        yoloView.sliderIoU.removeFromSuperview()
        yoloView.labelSliderIoU.isHidden = true
        yoloView.labelSliderIoU.removeFromSuperview()
        
        // Hide other controls
        yoloView.labelName.isHidden = true
        yoloView.labelName.removeFromSuperview()
        yoloView.labelFPS.isHidden = true
        yoloView.labelFPS.removeFromSuperview()
        yoloView.labelZoom.isHidden = true
        yoloView.activityIndicator.isHidden = true
        yoloView.playButton.isHidden = true
        yoloView.pauseButton.isHidden = true
        yoloView.switchCameraButton.isHidden = true
        yoloView.toolbar.isHidden = true
        yoloView.toolbar.removeFromSuperview()
        
        print("🟢 YOLOView controls hidden and sliders removed from view hierarchy")
        
        // Debug camera layer
        if let previewLayer = yoloView.layer.sublayers?.first(where: { $0 is AVCaptureVideoPreviewLayer }) {
            print("🟢 Found camera preview layer: \(previewLayer)")
            print("  - Frame: \(previewLayer.frame)")
            print("  - Hidden: \(previewLayer.isHidden)")
            print("  - Opacity: \(previewLayer.opacity)")
        } else {
            print("🔴 Camera preview layer not found!")
        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        print("🟢 ExternalViewController viewDidAppear")
        print("🟢 View bounds: \(view.bounds)")
        
        // Create and start YOLOView on external display
        if !self.isInitialized {
            self.isInitialized = true
            
            // Create YOLOView now that the view is fully ready
            print("🟢 Creating YOLOView for external display")
            print("🟢 View bounds at creation: \(view.bounds)")
            
            // Create YOLOView without model initially - will be set when main app notifies
            yoloView = YOLOView(frame: view.bounds)
            yoloView?.delegate = self
            yoloView?.backgroundColor = .clear // Make YOLOView background transparent
            
            if let yoloView = yoloView {
                // Add YOLOView first at the bottom
                view.insertSubview(yoloView, at: 0)
                
                // Add constraints for full screen display
                yoloView.translatesAutoresizingMaskIntoConstraints = false
                NSLayoutConstraint.activate([
                    yoloView.topAnchor.constraint(equalTo: view.topAnchor),
                    yoloView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
                    yoloView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
                    yoloView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
                ])
                
                // Force layout update
                view.layoutIfNeeded()
                
                print("🟢 YOLOView created with frame: \(yoloView.frame)")
                
                // Re-add UI elements on top of YOLOView to ensure they're visible
                // Add logo on top of YOLOView (camera feed)
                if let logoImageView = self.logoImageView {
                    logoImageView.removeFromSuperview()
                    view.addSubview(logoImageView)
                }
                if let labelName = self.labelName {
                    labelName.removeFromSuperview()
                    view.addSubview(labelName)
                }
                if let labelFPS = self.labelFPS {
                    labelFPS.removeFromSuperview()
                    view.addSubview(labelFPS)
                }
                // Don't re-add segmentedControl since it's hidden
                
                // Force another layout update
                view.layoutIfNeeded()
                
                // Debug: Print view hierarchy
                print("🔍 View hierarchy after setup:")
                for (index, subview) in view.subviews.enumerated() {
                    print("  \(index): \(type(of: subview))")
                }
                
                // Re-setup constraints since we re-added the views
                let scaleFactor = calculateScaleFactor(for: view.bounds.size)
                setupConstraints(scaleFactor: scaleFactor)
                
                // Debug UI elements
                print("📱 UI Elements Debug:")
                print("  - labelName frame: \(self.labelName.frame)")
                print("  - labelName font: \(self.labelName.font.pointSize)pt")
                print("  - labelName hidden: \(self.labelName.isHidden)")
                print("  - labelFPS frame: \(self.labelFPS.frame)")
                print("  - labelFPS font: \(self.labelFPS.font.pointSize)pt")
                print("  - segmentedControl frame: \(self.segmentedControl.frame)")
                print("  - segmentedControl hidden: \(self.segmentedControl.isHidden)")
                print("  - segmentedControl segments: \(self.segmentedControl.numberOfSegments)")
                
                // Wait a bit longer to ensure main YOLOView has released camera
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    self.hideYOLOViewControls()
                    
                    // Debug layers
                    print("🟢 YOLOView layers after creation:")
                    if let layers = yoloView.layer.sublayers {
                        for (index, layer) in layers.enumerated() {
                            print("  Layer \(index): \(type(of: layer))")
                            if layer is AVCaptureVideoPreviewLayer {
                                print("    ✅ Found AVCaptureVideoPreviewLayer!")
                                print("    Frame: \(layer.frame)")
                            }
                        }
                    }
                    
                    // Don't load initial model - wait for main app to notify us
                    print("🟢 External display ready, waiting for model from main app")
                    
                    // Notify main app
                    NotificationCenter.default.post(name: .externalDisplayReady, object: nil)
                }
            }
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
        print("🟡 handleModelChange called")
        print("  - Notification: \(notification)")
        print("  - UserInfo: \(notification.userInfo ?? [:])")
        
        guard let userInfo = notification.userInfo,
              let taskString = userInfo["task"] as? String,
              let modelName = userInfo["modelName"] as? String else {
            print("🔴 Missing model change information")
            return
        }
        
        let task: YOLOTask
        switch taskString {
        case "detect": task = .detect
        case "segment": task = .segment
        case "classify": task = .classify
        case "pose": task = .pose
        case "obb": task = .obb
        default: task = .detect
        }
        
        print("🟢 External display updating to model: \(modelName), task: \(taskString)")
        
        // Update the model on external display
        currentTask = task
        currentModelName = modelName
        
        // Update segmented control to match
        if let taskIndex = tasks.firstIndex(where: { $0.value == task }) {
            DispatchQueue.main.async {
                self.segmentedControl.selectedSegmentIndex = taskIndex
            }
        }
        
        // Update the YOLOView with new model
        yoloView?.setModel(modelPathOrName: modelName, task: task) { result in
            switch result {
            case .success():
                print("🟢 External display model updated successfully")
                // Update model name label
                DispatchQueue.main.async {
                    // Extract just the filename without extension
                    let modelDisplayName = (self.currentModelName as NSString).lastPathComponent
                    var nameWithoutExtension = (modelDisplayName as NSString).deletingPathExtension
                    
                    // Remove .mlmodelc extension if present
                    if nameWithoutExtension.hasSuffix(".mlmodelc") {
                        nameWithoutExtension = (nameWithoutExtension as NSString).deletingPathExtension
                    }
                    
                    // Replace "yolo" with "YOLO" (case insensitive)
                    let displayName = nameWithoutExtension.replacingOccurrences(of: "yolo", with: "YOLO", options: .caseInsensitive)
                    
                    self.labelName.text = displayName
                    print("🏷️ External display model name updated: \(displayName)")
                }
                // Force layer refresh after model change
                self.yoloView?.setNeedsDisplay()
                self.yoloView?.layoutIfNeeded()
            case .failure(let error):
                print("🔴 Failed to update external display model: \(error)")
            }
        }
    }
    
    @objc private func handleThresholdChange(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let conf = userInfo["conf"] as? Double,
              let iou = userInfo["iou"] as? Double,
              let maxItems = userInfo["maxItems"] as? Int,
              let yoloView = yoloView else {
            return
        }
        
        // Update YOLOView thresholds
        yoloView.sliderConf.value = Float(conf)
        yoloView.sliderIoU.value = Float(iou)
        yoloView.sliderNumItems.value = Float(maxItems)
        
        // Update slider labels
        yoloView.labelSliderConf.text = String(format: "%.2f Confidence Threshold", conf)
        yoloView.labelSliderIoU.text = String(format: "%.2f IoU Threshold", iou)
        
        // The threshold values will be applied when the next frame is processed
        // since YOLOView reads these slider values during inference
        
        print("📊 External display thresholds updated - Conf: \(conf), IoU: \(iou), Max items: \(maxItems)")
    }
    
    @objc private func handleTaskChange(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let taskName = userInfo["task"] as? String else {
            return
        }
        
        // Find the task in our tasks array and update segment control
        if let taskIndex = tasks.firstIndex(where: { $0.name == taskName }) {
            DispatchQueue.main.async {
                self.segmentedControl.selectedSegmentIndex = taskIndex
                print("📋 External display task changed to: \(taskName)")
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
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
        }
    }
    
    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        // Debug: Check if results are being received
        DispatchQueue.main.async {
            print("🟢 External display received result:")
            print("  - Task: \(self.currentTask)")
            print("  - Boxes: \(result.boxes.count)")
            
            switch self.currentTask {
            case .segment:
                print("  - Has masks: \(result.masks != nil)")
                if result.masks != nil {
                    print("  - Masks available")
                }
            case .pose:
                print("  - Keypoints: \(result.keypointsList.count)")
            case .classify:
                print("  - Classifications: \(result.boxes.count)")
            default:
                break
            }
            
            // Check YOLOView layers
            self.debugYOLOViewLayers()
        }
    }
    
    private func debugYOLOViewLayers() {
        guard let yoloView = yoloView else {
            print("🔴 YOLOView is nil")
            return
        }
        
        print("🔍 Checking YOLOView layers:")
        print("  - YOLOView bounds: \(yoloView.bounds)")
        print("  - YOLOView frame: \(yoloView.frame)")
        print("  - YOLOView layer sublayers: \(yoloView.layer.sublayers?.count ?? 0)")
        
        if let sublayers = yoloView.layer.sublayers {
            for (index, layer) in sublayers.enumerated() {
                print("    Layer \(index): \(type(of: layer)) - hidden: \(layer.isHidden), opacity: \(layer.opacity)")
                
                // Ensure result layers are visible (except UI label layers)
                if !(layer is CATextLayer) && layer.isHidden && index > 0 {
                    layer.isHidden = false
                    print("    🔧 Unhiding layer \(index)")
                }
            }
        }
    }
}


