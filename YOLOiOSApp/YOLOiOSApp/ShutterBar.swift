// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class ShutterBar: UIView {
    // UI Elements
    private let thumbnailButton = UIButton(type: .custom)
    private let shutterButton = UIButton(type: .custom)
    private let flipCameraButton = UIButton(type: .custom)
    
    // Constraints for orientation
    private var heightConstraint: NSLayoutConstraint?
    private var portraitConstraints: [NSLayoutConstraint] = []
    private var landscapeConstraints: [NSLayoutConstraint] = []
    
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
    
    override func layoutSubviews() {
        super.layoutSubviews()
        // Debug print to check if buttons are visible
        print("ShutterBar frame: \(frame)")
        print("Thumbnail button frame: \(thumbnailButton.frame)")
        print("Shutter button frame: \(shutterButton.frame)")
        print("Flip camera button frame: \(flipCameraButton.frame)")
    }
    
    func updateLayoutForOrientation(isLandscape: Bool) {
        NSLayoutConstraint.deactivate(portraitConstraints)
        NSLayoutConstraint.deactivate(landscapeConstraints)
        
        if isLandscape {
            NSLayoutConstraint.activate(landscapeConstraints)
        } else {
            NSLayoutConstraint.activate(portraitConstraints)
        }
        
        layoutIfNeeded()
    }
    
    private func setupUI() {
        backgroundColor = .ultralyticsSurfaceDark
        
        // Thumbnail Button - Modern design without border
        thumbnailButton.backgroundColor = .ultralyticsBrown
        thumbnailButton.layer.cornerRadius = 12
        thumbnailButton.layer.borderWidth = 0
        thumbnailButton.clipsToBounds = true
        thumbnailButton.contentMode = .scaleAspectFill
        thumbnailButton.addTarget(self, action: #selector(thumbnailTapped), for: .touchUpInside)
        
        // Shutter Button - Modern design with gray border and black gap
        shutterButton.backgroundColor = .black  // Black background for the gap
        shutterButton.layer.cornerRadius = 34
        shutterButton.layer.borderWidth = 3
        shutterButton.layer.borderColor = UIColor.gray.cgColor  // Gray border
        
        // Add inner white circle with gap from border
        let innerCircle = UIView()
        innerCircle.backgroundColor = .white
        innerCircle.layer.cornerRadius = 29  // Larger radius for smaller black gap
        innerCircle.isUserInteractionEnabled = false
        innerCircle.translatesAutoresizingMaskIntoConstraints = false
        shutterButton.addSubview(innerCircle)
        
        NSLayoutConstraint.activate([
            innerCircle.centerXAnchor.constraint(equalTo: shutterButton.centerXAnchor),
            innerCircle.centerYAnchor.constraint(equalTo: shutterButton.centerYAnchor),
            innerCircle.widthAnchor.constraint(equalToConstant: 58),  // Larger white circle for smaller gap
            innerCircle.heightAnchor.constraint(equalToConstant: 58)
        ])
        
        // Add shadow for depth
        shutterButton.layer.shadowColor = UIColor.black.cgColor
        shutterButton.layer.shadowOffset = CGSize(width: 0, height: 2)
        shutterButton.layer.shadowRadius = 4
        shutterButton.layer.shadowOpacity = 0.2
        
        let shutterTap = UITapGestureRecognizer(target: self, action: #selector(shutterTapped))
        let shutterLongPress = UILongPressGestureRecognizer(target: self, action: #selector(shutterLongPressed))
        shutterLongPress.minimumPressDuration = 0.7
        
        shutterButton.addGestureRecognizer(shutterTap)
        shutterButton.addGestureRecognizer(shutterLongPress)
        
        // Flip Camera Button - Modern circular design without border
        flipCameraButton.backgroundColor = UIColor.white.withAlphaComponent(0.1)
        flipCameraButton.setImage(UIImage(systemName: "arrow.triangle.2.circlepath"), for: .normal)
        flipCameraButton.tintColor = .white
        flipCameraButton.layer.cornerRadius = 22
        flipCameraButton.layer.borderWidth = 0
        flipCameraButton.addTarget(self, action: #selector(flipCameraTapped), for: .touchUpInside)
        
        // Layout
        [thumbnailButton, shutterButton, flipCameraButton].forEach {
            addSubview($0)
            $0.translatesAutoresizingMaskIntoConstraints = false
        }
        
        // Common constraints (button sizes)
        let commonConstraints = [
            thumbnailButton.widthAnchor.constraint(equalToConstant: 48),
            thumbnailButton.heightAnchor.constraint(equalToConstant: 48),
            shutterButton.widthAnchor.constraint(equalToConstant: 68),
            shutterButton.heightAnchor.constraint(equalToConstant: 68),
            flipCameraButton.widthAnchor.constraint(equalToConstant: 44),
            flipCameraButton.heightAnchor.constraint(equalToConstant: 44)
        ]
        
        // Portrait constraints - horizontal layout
        portraitConstraints = [
            thumbnailButton.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 24),
            thumbnailButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            
            shutterButton.centerXAnchor.constraint(equalTo: centerXAnchor),
            shutterButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            
            flipCameraButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -24),
            flipCameraButton.centerYAnchor.constraint(equalTo: centerYAnchor)
        ]
        
        // Landscape constraints - vertical layout
        landscapeConstraints = [
            thumbnailButton.centerXAnchor.constraint(equalTo: centerXAnchor),
            thumbnailButton.topAnchor.constraint(equalTo: topAnchor, constant: 24),
            
            shutterButton.centerXAnchor.constraint(equalTo: centerXAnchor),
            shutterButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            
            flipCameraButton.centerXAnchor.constraint(equalTo: centerXAnchor),
            flipCameraButton.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -24)
        ]
        
        // Activate common constraints
        NSLayoutConstraint.activate(commonConstraints)
        
        // Default to portrait
        NSLayoutConstraint.activate(portraitConstraints)
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
            // Start recording animation - change inner circle to red
            if let innerCircle = self.shutterButton.subviews.first {
                UIView.animate(withDuration: 0.3) {
                    innerCircle.backgroundColor = .red
                }
            }
        } else if gesture.state == .ended || gesture.state == .cancelled {
            // Reset color when recording stops
            if let innerCircle = self.shutterButton.subviews.first {
                UIView.animate(withDuration: 0.3) {
                    innerCircle.backgroundColor = .white
                }
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
        // Change inner circle color, not button background
        if let innerCircle = shutterButton.subviews.first {
            innerCircle.backgroundColor = isRecording ? .red : .white
        }
    }
    
}