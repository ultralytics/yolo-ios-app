import UIKit
import PhotosUI
import YOLO

class ViewController: UIViewController, PHPickerViewControllerDelegate {
    
    var model:YOLO!
    var imageView:UIImageView!
    var pickButton:UIButton!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        model = YOLO("yolo11n", task: .detect)
        
        setupView()
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let result = results.first else { return }
        if result.itemProvider.canLoadObject(ofClass: UIImage.self) {
            result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] image, error  in
                if let image = image as? UIImage,  let safeSelf = self {
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
    
    @objc func presentPhPicker(){
        var configuration = PHPickerConfiguration()
        configuration.selectionLimit = 1
        configuration.filter = .images
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = self
        self.present(picker, animated: true)
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
    
    private func setupView(){
        imageView = UIImageView(frame: view.bounds)
        view.addSubview(imageView)
        imageView.contentMode = .scaleAspectFit
        
        pickButton = UIButton(frame: CGRect(x: view.center.x - 50, y: view.bounds.maxY-75, width: 100, height: 50))
        pickButton.setTitle("Pick Image", for: .normal)
        pickButton.addTarget(self, action: #selector(presentPhPicker), for: .touchUpInside)
        pickButton.backgroundColor = .systemBlue
        view.addSubview(pickButton)
    }
}



