import SwiftUI
import PhotosUI
import YOLO

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
            }
            else if let input = inputImage {
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

func getCorrectOrientationUIImage(uiImage:UIImage) -> UIImage {
    var newImage = UIImage()
    let ciContext = CIContext()
    switch uiImage.imageOrientation.rawValue {
    case 1:
        guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.down),
              let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent) else { return uiImage}
        
        newImage = UIImage(cgImage: cgImage)
    case 3:
        guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.right),
              let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent) else { return uiImage}
        newImage = UIImage(cgImage: cgImage)
    default:
        newImage = uiImage
    }
    return newImage
}
