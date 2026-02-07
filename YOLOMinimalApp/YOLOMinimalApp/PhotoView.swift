// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import PhotosUI
import SwiftUI
import YOLOUI

/// Single-image inference view with photo picker.
struct PhotoView: View {
  @State private var selectedItem: PhotosPickerItem?
  @State private var resultImage: CGImage?
  @State private var isLoading = false

  var body: some View {
    VStack {
      if let resultImage {
        Image(decorative: resultImage, scale: 1.0)
          .resizable()
          .scaledToFit()
      } else {
        ContentUnavailableView("No Image", systemImage: "photo", description: Text("Pick a photo to analyze"))
      }

      PhotosPicker(selection: $selectedItem, matching: .images) {
        Label("Pick Photo", systemImage: "photo.on.rectangle")
          .padding()
          .foregroundStyle(.white)
          .background(Color.accentColor)
          .clipShape(RoundedRectangle(cornerRadius: 8))
      }
      .disabled(isLoading)
      .onChange(of: selectedItem) { _, newItem in
        guard let newItem else { return }
        Task { await processImage(item: newItem) }
      }

      if isLoading {
        ProgressView("Running inference...")
      }
    }
    .padding()
  }

  private func processImage(item: PhotosPickerItem) async {
    isLoading = true
    defer { isLoading = false }

    guard let data = try? await item.loadTransferable(type: Data.self),
      let uiImage = UIImage(data: data),
      let cgImage = uiImage.cgImage
    else { return }

    do {
      let model = try await YOLO("yolo26n", task: .detect)
      let result = model(cgImage)
      resultImage = result.annotatedImage ?? cgImage
    } catch {
      resultImage = cgImage
    }
  }
}
