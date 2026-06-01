// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing a SwiftUI example for single image
//  object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://www.ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ContentView shows how to run static image analysis with YOLO models in SwiftUI. It provides a user interface
//  with a PhotosPicker for selecting images from the device's photo library and displays both the original and
//  processed images with detection results. The example covers initializing a YOLO detection model, loading images from
//  the photo picker, correcting image orientation, and running inference on the selected image. The pattern can be
//  adapted for other task types by changing the task parameter and model name during initialization.

import PhotosUI
import SwiftUI
import YOLO

/// A SwiftUI view that demonstrates single-image object detection using the YOLO framework.
struct ContentView: View {
  @State private var selectedItem: PhotosPickerItem?
  @State private var inputImage: UIImage?
  @State private var yoloResult: YOLOResult?

  let yolo = YOLO("yolo26n", task: .detect)

  var body: some View {
    VStack {
      if let annotated = yoloResult?.annotatedImage {
        Image(uiImage: annotated)
          .resizable()
          .scaledToFit()
      } else if let input = inputImage {
        Image(uiImage: input)
          .resizable()
          .scaledToFit()
      } else {
        Text("No image selected")
      }

      PhotosPicker(
        selection: $selectedItem,
        matching: .images,
        photoLibrary: .shared()
      ) {
        Text("Pick Photo")
          .padding()
          .foregroundColor(.white)
          .background(Color.blue)
          .cornerRadius(8)
      }
      .onChange(of: selectedItem) { newItem in
        Task {
          guard let newItem = newItem,
            let data = try? await newItem.loadTransferable(type: Data.self),
            let uiImage = UIImage(data: data)
          else { return }
          let correctOrientationUIImage = getCorrectOrientationUIImage(uiImage: uiImage)
          inputImage = correctOrientationUIImage

          yoloResult = yolo(correctOrientationUIImage)
        }
      }
    }
    .padding()
  }
}

/// Corrects the orientation of an image for processing by the YOLO model.
///
/// Images from the photo library may carry orientation metadata rather than correctly oriented pixels. This function
/// applies the orientation so the returned image has its pixels in the correct order for inference.
///
/// - Parameter uiImage: The input image that may have incorrect orientation metadata.
/// - Returns: A new UIImage with the correct orientation for processing.
func getCorrectOrientationUIImage(uiImage: UIImage) -> UIImage {
  var newImage = UIImage()
  let ciContext = CIContext()
  switch uiImage.imageOrientation.rawValue {
  case 1:
    guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.down),
      let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent)
    else { return uiImage }

    newImage = UIImage(cgImage: cgImage)
  case 3:
    guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.right),
      let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent)
    else { return uiImage }
    newImage = UIImage(cgImage: cgImage)
  default:
    newImage = uiImage
  }
  return newImage
}
