// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class ShutterBar: UIView {
    // UI Elements
    private let thumbnailButton = UIButton(type: .custom)
    private let shutterButton = UIButton(type: .custom)
    private let flipCameraButton = UIButton(type: .custom)
    
    // Callbacks
    var onThumbnailTap: (() -> Void)?
    var onShutterTap: (() -> Void)?
    var onShutterLongPress: (() -> Void)?
    var onFlipCamera: (() -> Void)?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupUI()
    }
    
    private func setupUI() {
        backgroundColor = .ultralyticsSurfaceDark
        
        // Thumbnail Button
        thumbnailButton.backgroundColor = .ultralyticsBrown
        thumbnailButton.layer.cornerRadius = 8
        thumbnailButton.clipsToBounds = true
        thumbnailButton.addTarget(self, action: #selector(thumbnailTapped), for: .touchUpInside)
        
        // Shutter Button
        shutterButton.backgroundColor = .white
        shutterButton.layer.cornerRadius = 34
        shutterButton.layer.borderWidth = 4
        shutterButton.layer.borderColor = UIColor.black.cgColor
        
        let shutterTap = UITapGestureRecognizer(target: self, action: #selector(shutterTapped))
        let shutterLongPress = UILongPressGestureRecognizer(target: self, action: #selector(shutterLongPressed))
        shutterLongPress.minimumPressDuration = 0.7
        
        shutterButton.addGestureRecognizer(shutterTap)
        shutterButton.addGestureRecognizer(shutterLongPress)
        
        // Flip Camera Button
        flipCameraButton.backgroundColor = .clear
        flipCameraButton.setImage(UIImage(systemName: "camera.rotate"), for: .normal)
        flipCameraButton.tintColor = .white
        flipCameraButton.addTarget(self, action: #selector(flipCameraTapped), for: .touchUpInside)
        
        // Layout
        [thumbnailButton, shutterButton, flipCameraButton].forEach {
            addSubview($0)
            $0.translatesAutoresizingMaskIntoConstraints = false
        }
        
        NSLayoutConstraint.activate([
            // Container height
            heightAnchor.constraint(equalToConstant: 96),
            
            // Thumbnail
            thumbnailButton.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 24),
            thumbnailButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            thumbnailButton.widthAnchor.constraint(equalToConstant: 48),
            thumbnailButton.heightAnchor.constraint(equalToConstant: 48),
            
            // Shutter
            shutterButton.centerXAnchor.constraint(equalTo: centerXAnchor),
            shutterButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            shutterButton.widthAnchor.constraint(equalToConstant: 68),
            shutterButton.heightAnchor.constraint(equalToConstant: 68),
            
            // Flip Camera
            flipCameraButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -24),
            flipCameraButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            flipCameraButton.widthAnchor.constraint(equalToConstant: 44),
            flipCameraButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    @objc private func thumbnailTapped() {
        onThumbnailTap?()
    }
    
    @objc private func shutterTapped() {
        // Flash animation
        let flashView = UIView(frame: UIScreen.main.bounds)
        flashView.backgroundColor = .white
        flashView.alpha = 0
        
        if let window = UIApplication.shared.windows.first(where: { $0.isKeyWindow }) {
            window.addSubview(flashView)
            
            UIView.animate(withDuration: 0.08, animations: {
                flashView.alpha = 0.8
            }) { _ in
                flashView.removeFromSuperview()
            }
        }
        
        onShutterTap?()
    }
    
    @objc private func shutterLongPressed(_ gesture: UILongPressGestureRecognizer) {
        if gesture.state == .began {
            onShutterLongPress?()
            // Start recording animation
            UIView.animate(withDuration: 0.3) {
                self.shutterButton.backgroundColor = .red
            }
        } else if gesture.state == .ended || gesture.state == .cancelled {
            // Reset color when recording stops
            UIView.animate(withDuration: 0.3) {
                self.shutterButton.backgroundColor = .white
            }
        }
    }
    
    @objc private func flipCameraTapped() {
        onFlipCamera?()
    }
    
    func updateThumbnail(_ image: UIImage?) {
        thumbnailButton.setImage(image, for: .normal)
        thumbnailButton.imageView?.contentMode = .scaleAspectFill
    }
    
    func setRecording(_ isRecording: Bool) {
        shutterButton.backgroundColor = isRecording ? .red : .white
    }
}