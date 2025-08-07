// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing a UIKit example for single image object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController demonstrates how to implement static image analysis using YOLO models in UIKit.
//  It provides a user interface for selecting images from the device's photo library and displays
//  both the original and processed images with detection results. The example shows how to initialize
//  a YOLO model for segmentation, handle image orientation correction, and run inference on selected images.

import PhotosUI
import UIKit
import YOLO

/// A view controller that demonstrates YOLO model inference on a single image using UIKit.
///
/// This view controller allows users to select an image from their photo library and performs
/// YOLO model inference on the selected image. It uses a segmentation model by default but can be
/// configured to use other YOLO task types like detection, classification, or pose estimation.
///
/// - Note: This example uses the PhotosUI framework for image selection and requires photo library access.
/// - Important: The app requires at least iOS 16.0 or higher to run.
class ViewController: UIViewController, PHPickerViewControllerDelegate {

  /// The YOLO model instance used for inference.
  var model: YOLO!

  /// The image view that displays the original and annotated images.
  var imageView: UIImageView!

  /// The button that triggers the photo picker interface.
  var pickButton: UIButton!

  /// Sets up the view and initializes the YOLO model.
  override func viewDidLoad() {
    super.viewDidLoad()
    //      view.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(presentPhPicker)))
    // Initialize YOLO model with a segmentation task
    // You can change the model or task type to use detection, classification, etc.
    model = YOLO("yolo11n", task: .detect) { [self] result in
      switch result {
      case .success(_):
        print("predictor initialized")
        setupView()
      case .failure(let error):
        fatalError(error.localizedDescription)
      }
    }
  }

  /// Handles the result from the photo picker and performs YOLO inference on the selected image.
  ///
  /// - Parameters:
  ///   - picker: The photo picker view controller.
  ///   - results: The array of selected photo picker results.
  func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
    picker.dismiss(animated: true)
    guard let result = results.first else { return }
    if result.itemProvider.canLoadObject(ofClass: UIImage.self) {
      result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] image, error in
        if let image = image as? UIImage, let safeSelf = self {
          let correctOrientImage = safeSelf.getCorrectOrientationUIImage(uiImage: image)
          let date = Date()
          let result = safeSelf.model(correctOrientImage)
          let time = Date().timeIntervalSince(date)
          print(result)
          DispatchQueue.main.async {
            UIImageWriteToSavedPhotosAlbum(result.annotatedImage!, self, nil, nil)
            safeSelf.imageView.image = result.annotatedImage
          }
        }
      }
    }
  }

  /// Presents the photo picker interface to select an image.
  @objc func presentPhPicker() {
    var configuration = PHPickerConfiguration()
    configuration.selectionLimit = 1
    configuration.filter = .images
    let picker = PHPickerViewController(configuration: configuration)
    picker.delegate = self
    self.present(picker, animated: true)
  }

  /// Corrects the orientation of the image to ensure proper processing by the YOLO model.
  ///
  /// - Parameter uiImage: The input image that may have incorrect orientation metadata.
  /// - Returns: A UIImage with the correct orientation for processing.
  func getCorrectOrientationUIImage(uiImage: UIImage) -> UIImage {
    var newImage = UIImage()
    let ciContext = CIContext()
    switch uiImage.imageOrientation.rawValue {
    case 1:
      guard
        let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.down),
        let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent)
      else { return uiImage }

      newImage = UIImage(cgImage: cgImage)
    case 3:
      guard
        let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.right),
        let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent)
      else { return uiImage }
      newImage = UIImage(cgImage: cgImage)
    default:
      newImage = uiImage
    }
    return newImage
  }

  /// Sets up the UI components including the image view and pick button.
  private func setupView() {
    imageView = UIImageView(frame: view.bounds)
    view.addSubview(imageView)
    imageView.contentMode = .scaleAspectFit

    pickButton = UIButton(
      frame: CGRect(x: view.center.x - 50, y: view.bounds.maxY - 100, width: 100, height: 50))
    pickButton.setTitle("Pick Image", for: .normal)
    pickButton.addTarget(self, action: #selector(presentPhPicker), for: .touchUpInside)
    pickButton.backgroundColor = .systemBlue
    pickButton.clipsToBounds = true
    pickButton.layer.cornerRadius = 8
    view.addSubview(pickButton)
  }
}
