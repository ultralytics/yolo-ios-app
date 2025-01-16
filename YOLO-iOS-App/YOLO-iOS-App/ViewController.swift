//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  This file is part of the Ultralytics YOLO app, enabling real-time object detection using YOLO11 models on iOS devices.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This ViewController manages the app's main screen, handling video capture, model selection, detection visualization,
//  and user interactions. It sets up and controls the video preview layer, handles model switching via a segmented control,
//  manages UI elements like sliders for confidence and IoU thresholds, and displays detection results on the video feed.
//  It leverages CoreML, Vision, and AVFoundation frameworks to perform real-time object detection and to interface with
//  the device's camera.

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

    override func viewDidLoad() {
        super.viewDidLoad()
        yoloView.setModel(modelPathOrName: "yolo11n", task: .detect)
        setupShareButton()
    }
        
    @IBAction func vibrate(_ sender: Any) {
        selection.selectionChanged()
    }
    
    @IBAction func indexChanged(_ sender: Any) {
        selection.selectionChanged()
        
        /// Switch model
        switch segmentedControl.selectedSegmentIndex {
        case 0:
            yoloView.setModel(modelPathOrName: "yolo11n", task: .detect)
        case 1:
            yoloView.setModel(modelPathOrName: "yolo11s", task: .detect)
        case 2:
            yoloView.setModel(modelPathOrName: "yolo11m", task: .detect)
        case 3:
            yoloView.setModel(modelPathOrName: "yolo11l", task: .detect)
        case 4:
            yoloView.setModel(modelPathOrName: "yolo11x", task: .detect)
        default:
            break
        }
    }
        
    @IBAction func logoButton(_ sender: Any) {
        selection.selectionChanged()
        if let link = URL(string: "https://www.ultralytics.com") {
            UIApplication.shared.open(link)
        }
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
        shareButton.frame = CGRect(x: view.bounds.maxX - 49.5, y: view.bounds.maxY - 66, width: 49.5, height: 49.5)
    }
    
    @objc func shareButtonTapped() {
        selection.selectionChanged()
        yoloView.capturePhoto { [weak self] captured in
            guard let self = self else { return }
            if let image = captured {
                DispatchQueue.main.async {
                    let activityViewController = UIActivityViewController(
                      activityItems: [image], applicationActivities: nil)
                    activityViewController.popoverPresentationController?.sourceView = self.View0
                    self.present(activityViewController, animated: true, completion: nil)

                }
            } else {
                print("error capturing photo")
            }
        }
    }
}
