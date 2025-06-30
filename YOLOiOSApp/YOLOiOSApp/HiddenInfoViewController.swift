// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class HiddenInfoViewController: UIViewController {
    
    private let logoImageView = UIImageView()
    private let captionLabel = UILabel()
    private let disclaimerLabel = UILabel()
    private let closeButton = UIButton(type: .system)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    private func setupUI() {
        view.backgroundColor = .white
        
        // Logo
        logoImageView.image = UIImage(named: "ultralytics_icon")
        logoImageView.contentMode = .scaleAspectFit
        
        // Caption
        captionLabel.text = "In order to test your models you need to use Hub App"
        captionLabel.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        captionLabel.textColor = .black
        captionLabel.textAlignment = .center
        captionLabel.numberOfLines = 0
        
        // Disclaimer
        disclaimerLabel.text = "Ultralytics YOLO iOS App v1.0"
        disclaimerLabel.font = UIFont.systemFont(ofSize: 12, weight: .regular)
        disclaimerLabel.textColor = .systemGray
        disclaimerLabel.textAlignment = .center
        
        // Close button
        closeButton.setTitle("Close", for: .normal)
        closeButton.titleLabel?.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        closeButton.addTarget(self, action: #selector(closeTapped), for: .touchUpInside)
        
        // Layout
        [logoImageView, captionLabel, disclaimerLabel, closeButton].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview($0)
        }
        
        NSLayoutConstraint.activate([
            // Logo
            logoImageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            logoImageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 60),
            logoImageView.widthAnchor.constraint(equalToConstant: 80),
            logoImageView.heightAnchor.constraint(equalToConstant: 80),
            
            // Caption
            captionLabel.topAnchor.constraint(equalTo: logoImageView.bottomAnchor, constant: 30),
            captionLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
            captionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),
            
            // Disclaimer
            disclaimerLabel.bottomAnchor.constraint(equalTo: closeButton.topAnchor, constant: -20),
            disclaimerLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            
            // Close button
            closeButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            closeButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -40),
            closeButton.widthAnchor.constraint(equalToConstant: 100),
            closeButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    @objc private func closeTapped() {
        dismiss(animated: true, completion: nil)
    }
}