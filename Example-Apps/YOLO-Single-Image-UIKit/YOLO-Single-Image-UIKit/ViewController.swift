//
//  ViewController.swift
//  YOLO-Single-Image-UIKit
//
//  Created by Ultralytics
//  License: MIT
//

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

    // Initialize YOLO model with a segmentation task
    // You can change the model or task type to use detection, classification, etc.
    model = YOLO("yolo11x-seg", task: .segment) { [self] result in
      switch result {
      case .success(_):
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
