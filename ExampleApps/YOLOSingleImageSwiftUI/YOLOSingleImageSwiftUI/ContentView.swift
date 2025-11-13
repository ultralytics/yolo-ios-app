// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing a SwiftUI example for single image object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ContentView demonstrates how to implement static image analysis using YOLO models in SwiftUI.
//  It provides a user interface with a PhotosPicker for selecting images from the device's photo library
//  and displays both the original and processed images with detection results. The example shows how to
//  initialize a YOLO model for segmentation, load images from the photo picker, handle image orientation
//  correction, and run inference on the selected image. This pattern can be adapted for other model
//  types by changing the task parameter and model name during initialization.

import PhotosUI
import SwiftUI
import YOLO

/// A SwiftUI view that demonstrates single-image object detection using the YOLO framework.
struct ContentView: View {
  @State private var selectedItem: PhotosPickerItem?
  @State private var inputImage: UIImage?
  @State private var yoloResult: YOLOResult?

  let yolo = YOLO("yolo11n", task: .detect)

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
          // Image orientation is automatically normalized by the YOLO package
          inputImage = uiImage
          yoloResult = yolo(uiImage)
        }
      }
    }
    .padding()
  }
}
