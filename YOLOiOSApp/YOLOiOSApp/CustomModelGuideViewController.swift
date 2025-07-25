// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit
import SafariServices

class CustomModelGuideViewController: UIViewController {
    
    // MARK: - UI Components
    
    private let scrollView = UIScrollView()
    private let contentView = UIView()
    
    private let closeButton: UIButton = {
        let button = UIButton(type: .system)
        button.setImage(UIImage(systemName: "xmark.circle.fill"), for: .normal)
        button.tintColor = .white
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    
    private let descriptionLabel: UILabel = {
        let label = UILabel()
        label.text = "To create and use custom models on mobile, train your custom model with Ultralytics HUB and test it with the Ultralytics HUB App."
        label.font = UIFont.systemFont(ofSize: 17, weight: .regular)
        label.textColor = .white
        label.textAlignment = .center
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let hubLogoImageView: UIImageView = {
        let imageView = UIImageView()
        imageView.image = UIImage(named: "Ultralytics_HUB")
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.isUserInteractionEnabled = true
        return imageView
    }()
    
    private let appStoreImageView: UIImageView = {
        let imageView = UIImageView()
        imageView.image = UIImage(named: "app-store")
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.isUserInteractionEnabled = true
        return imageView
    }()
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupGestures()
    }
    
    // MARK: - UI Setup
    
    private func setupUI() {
        // Background gradient
        let gradientLayer = CAGradientLayer()
        gradientLayer.colors = [
            UIColor(red: 0.1, green: 0.1, blue: 0.15, alpha: 1.0).cgColor,
            UIColor.black.cgColor
        ]
        gradientLayer.locations = [0.0, 1.0]
        gradientLayer.frame = view.bounds
        view.layer.insertSublayer(gradientLayer, at: 0)
        
        // Scroll view setup
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.showsVerticalScrollIndicator = false
        contentView.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(scrollView)
        scrollView.addSubview(contentView)
        
        // Add components to content view
        contentView.addSubview(descriptionLabel)
        contentView.addSubview(hubLogoImageView)
        contentView.addSubview(appStoreImageView)
        
        // Add close button to main view
        view.addSubview(closeButton)
        
        // Setup constraints
        NSLayoutConstraint.activate([
            // Close button
            closeButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            closeButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            closeButton.widthAnchor.constraint(equalToConstant: 32),
            closeButton.heightAnchor.constraint(equalToConstant: 32),
            
            // Scroll view
            scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Content view
            contentView.topAnchor.constraint(equalTo: scrollView.topAnchor),
            contentView.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor),
            contentView.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor),
            contentView.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor),
            contentView.widthAnchor.constraint(equalTo: scrollView.widthAnchor),
            
            // Description label - now at the top
            descriptionLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 100),
            descriptionLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 32),
            descriptionLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -32),
            
            // HUB logo
            hubLogoImageView.topAnchor.constraint(equalTo: descriptionLabel.bottomAnchor, constant: 48),
            hubLogoImageView.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            hubLogoImageView.widthAnchor.constraint(equalToConstant: 240),
            hubLogoImageView.heightAnchor.constraint(equalToConstant: 60),
            
            // App Store button
            appStoreImageView.topAnchor.constraint(equalTo: hubLogoImageView.bottomAnchor, constant: 48),
            appStoreImageView.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            appStoreImageView.widthAnchor.constraint(equalToConstant: 180),
            appStoreImageView.heightAnchor.constraint(equalToConstant: 60),
            appStoreImageView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -80)
        ])
        
        // Add subtle animations
        addHoverEffect(to: hubLogoImageView)
        addHoverEffect(to: appStoreImageView)
    }
    
    private func addHoverEffect(to imageView: UIImageView) {
        imageView.layer.shadowColor = UIColor.ultralyticsLime.cgColor
        imageView.layer.shadowRadius = 0
        imageView.layer.shadowOpacity = 0
        imageView.layer.shadowOffset = CGSize(width: 0, height: 2)
    }
    
    // MARK: - Gestures
    
    private func setupGestures() {
        // Close button
        closeButton.addTarget(self, action: #selector(closeButtonTapped), for: .touchUpInside)
        
        // HUB logo tap
        let hubTap = UITapGestureRecognizer(target: self, action: #selector(hubTapped))
        hubLogoImageView.addGestureRecognizer(hubTap)
        
        // App Store tap
        let appStoreTap = UITapGestureRecognizer(target: self, action: #selector(appStoreTapped))
        appStoreImageView.addGestureRecognizer(appStoreTap)
    }
    
    // MARK: - Actions
    
    @objc private func closeButtonTapped() {
        dismiss(animated: true)
    }
    
    @objc private func hubTapped() {
        animateTap(hubLogoImageView) {
            self.openURL("https://hub.ultralytics.com/")
        }
    }
    
    @objc private func appStoreTapped() {
        animateTap(appStoreImageView) {
            self.openURL("https://apps.apple.com/en/app/ultralytics-hub/id1583935240")
        }
    }
    
    private func openURL(_ urlString: String) {
        guard let url = URL(string: urlString) else { return }
        
        let config = SFSafariViewController.Configuration()
        config.entersReaderIfAvailable = false
        
        let safariVC = SFSafariViewController(url: url, configuration: config)
        safariVC.preferredControlTintColor = .ultralyticsLime
        present(safariVC, animated: true)
    }
    
    private func animateTap(_ view: UIView, completion: @escaping () -> Void) {
        UIView.animate(withDuration: 0.1, animations: {
            view.transform = CGAffineTransform(scaleX: 0.95, y: 0.95)
            view.layer.shadowRadius = 10
            view.layer.shadowOpacity = 0.3
        }) { _ in
            UIView.animate(withDuration: 0.1, animations: {
                view.transform = .identity
                view.layer.shadowRadius = 0
                view.layer.shadowOpacity = 0
            }) { _ in
                completion()
            }
        }
    }
    
    // MARK: - Layout
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        // Update gradient frame
        if let gradientLayer = view.layer.sublayers?.first as? CAGradientLayer {
            gradientLayer.frame = view.bounds
        }
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
}