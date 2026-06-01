// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, providing an in-app information sheet with official resources.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import SafariServices
import UIKit

final class YOLOInfoViewController: UIViewController {
  private struct Resource {
    let title: String
    let subtitle: String
    let systemImage: String
    let url: URL
  }

  private let resources: [Resource] = [
    Resource(
      title: "Ultralytics Docs",
      subtitle: "Training, prediction, Core ML export, deployment, and licensing guides.",
      systemImage: "book",
      url: URL(string: "https://docs.ultralytics.com")!
    ),
    Resource(
      title: "YOLO Models",
      subtitle: "Compare model families, supported tasks, sizes, and performance.",
      systemImage: "square.stack.3d.up",
      url: URL(string: "https://docs.ultralytics.com/models/")!
    ),
    Resource(
      title: "Ultralytics Platform",
      subtitle: "Manage datasets, train models, and deploy computer vision workflows.",
      systemImage: "rectangle.3.group",
      url: URL(string: "https://platform.ultralytics.com")!
    ),
    Resource(
      title: "GitHub",
      subtitle: "Explore the iOS app source, Swift package, examples, and issues.",
      systemImage: "chevron.left.forwardslash.chevron.right",
      url: URL(string: "https://github.com/ultralytics/yolo-ios-app")!
    ),
    Resource(
      title: "Community",
      subtitle: "Ask questions and connect with the Ultralytics team and users.",
      systemImage: "message",
      url: URL(string: "https://discord.com/invite/ultralytics")!
    ),
  ]

  override func viewDidLoad() {
    super.viewDidLoad()

    title = "About YOLO"
    view.backgroundColor = .systemBackground
    navigationItem.rightBarButtonItem = UIBarButtonItem(
      systemItem: .done,
      primaryAction: UIAction { [weak self] _ in
        self?.dismiss(animated: true)
      })

    let scrollView = UIScrollView()
    scrollView.alwaysBounceVertical = true
    scrollView.translatesAutoresizingMaskIntoConstraints = false

    let stackView = UIStackView()
    stackView.axis = .vertical
    stackView.spacing = 18
    stackView.translatesAutoresizingMaskIntoConstraints = false

    view.addSubview(scrollView)
    scrollView.addSubview(stackView)

    NSLayoutConstraint.activate([
      scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
      scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
      scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      scrollView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
      stackView.topAnchor.constraint(
        equalTo: scrollView.contentLayoutGuide.topAnchor, constant: 20),
      stackView.leadingAnchor.constraint(
        equalTo: scrollView.frameLayoutGuide.leadingAnchor, constant: 20),
      stackView.trailingAnchor.constraint(
        equalTo: scrollView.frameLayoutGuide.trailingAnchor, constant: -20),
      stackView.bottomAnchor.constraint(
        equalTo: scrollView.contentLayoutGuide.bottomAnchor, constant: -28),
    ])

    stackView.addArrangedSubview(headerView())
    stackView.addArrangedSubview(
      infoSection(
        title: "The App",
        body:
          "Ultralytics YOLO runs real-time computer vision on your iPhone or iPad with Core ML. Use it to try detection, segmentation, classification, pose estimation, and oriented bounding box models directly on-device."
      ))
    stackView.addArrangedSubview(
      infoSection(
        title: "YOLO Models",
        body:
          "YOLO models are fast vision models built for practical inference. This app includes nano models for each task and can download larger official models when selected, so you can compare speed and detail on your own device."
      ))
    stackView.addArrangedSubview(resourcesSection())
  }

  private func headerView() -> UIView {
    let stackView = UIStackView()
    stackView.axis = .vertical
    stackView.alignment = .center
    stackView.spacing = 8

    let iconView = UIImageView(image: UIImage(systemName: "camera.viewfinder"))
    iconView.tintColor = .systemBlue
    iconView.contentMode = .scaleAspectFit
    iconView.translatesAutoresizingMaskIntoConstraints = false

    let titleLabel = UILabel()
    titleLabel.text = "Ultralytics YOLO"
    titleLabel.font = .preferredFont(forTextStyle: .title1)
    titleLabel.adjustsFontForContentSizeCategory = true
    titleLabel.textAlignment = .center

    let subtitleLabel = UILabel()
    subtitleLabel.text = "Real-time AI vision for iOS"
    subtitleLabel.font = .preferredFont(forTextStyle: .subheadline)
    subtitleLabel.adjustsFontForContentSizeCategory = true
    subtitleLabel.textColor = .secondaryLabel
    subtitleLabel.textAlignment = .center

    stackView.addArrangedSubview(iconView)
    stackView.addArrangedSubview(titleLabel)
    stackView.addArrangedSubview(subtitleLabel)

    NSLayoutConstraint.activate([
      iconView.widthAnchor.constraint(equalToConstant: 44),
      iconView.heightAnchor.constraint(equalToConstant: 44),
    ])

    return stackView
  }

  private func infoSection(title: String, body: String) -> UIView {
    let stackView = UIStackView()
    stackView.axis = .vertical
    stackView.spacing = 6

    let titleLabel = UILabel()
    titleLabel.text = title
    titleLabel.font = .preferredFont(forTextStyle: .headline)
    titleLabel.adjustsFontForContentSizeCategory = true

    let bodyLabel = UILabel()
    bodyLabel.text = body
    bodyLabel.font = .preferredFont(forTextStyle: .body)
    bodyLabel.adjustsFontForContentSizeCategory = true
    bodyLabel.textColor = .secondaryLabel
    bodyLabel.numberOfLines = 0

    stackView.addArrangedSubview(titleLabel)
    stackView.addArrangedSubview(bodyLabel)
    return stackView
  }

  private func resourcesSection() -> UIView {
    let stackView = UIStackView()
    stackView.axis = .vertical
    stackView.spacing = 10

    let titleLabel = UILabel()
    titleLabel.text = "Continue Learning"
    titleLabel.font = .preferredFont(forTextStyle: .headline)
    titleLabel.adjustsFontForContentSizeCategory = true
    stackView.addArrangedSubview(titleLabel)

    for resource in resources {
      stackView.addArrangedSubview(resourceButton(for: resource))
    }

    return stackView
  }

  private func resourceButton(for resource: Resource) -> UIButton {
    let action = UIAction { [weak self] _ in
      self?.open(resource.url)
    }
    let button = UIButton(type: .system, primaryAction: action)
    button.backgroundColor = .secondarySystemGroupedBackground
    button.layer.cornerRadius = 8
    button.accessibilityHint = "Opens \(resource.title)"

    let iconView = UIImageView(image: UIImage(systemName: resource.systemImage))
    iconView.tintColor = .systemBlue
    iconView.contentMode = .scaleAspectFit
    iconView.translatesAutoresizingMaskIntoConstraints = false

    let titleLabel = UILabel()
    titleLabel.text = resource.title
    titleLabel.font = .preferredFont(forTextStyle: .headline)
    titleLabel.adjustsFontForContentSizeCategory = true
    titleLabel.textColor = .systemBlue

    let subtitleLabel = UILabel()
    subtitleLabel.text = resource.subtitle
    subtitleLabel.font = .preferredFont(forTextStyle: .footnote)
    subtitleLabel.adjustsFontForContentSizeCategory = true
    subtitleLabel.textColor = .secondaryLabel
    subtitleLabel.numberOfLines = 0

    let textStackView = UIStackView(arrangedSubviews: [titleLabel, subtitleLabel])
    textStackView.axis = .vertical
    textStackView.spacing = 2

    let externalLinkView = UIImageView(image: UIImage(systemName: "arrow.up.forward"))
    externalLinkView.tintColor = .tertiaryLabel
    externalLinkView.contentMode = .scaleAspectFit
    externalLinkView.translatesAutoresizingMaskIntoConstraints = false

    let rowStackView = UIStackView(arrangedSubviews: [iconView, textStackView, externalLinkView])
    rowStackView.alignment = .center
    rowStackView.spacing = 12
    rowStackView.isUserInteractionEnabled = false
    rowStackView.translatesAutoresizingMaskIntoConstraints = false

    button.addSubview(rowStackView)
    NSLayoutConstraint.activate([
      iconView.widthAnchor.constraint(equalToConstant: 24),
      iconView.heightAnchor.constraint(equalToConstant: 24),
      externalLinkView.widthAnchor.constraint(equalToConstant: 16),
      externalLinkView.heightAnchor.constraint(equalToConstant: 16),
      rowStackView.topAnchor.constraint(equalTo: button.topAnchor, constant: 12),
      rowStackView.leadingAnchor.constraint(equalTo: button.leadingAnchor, constant: 12),
      rowStackView.trailingAnchor.constraint(equalTo: button.trailingAnchor, constant: -12),
      rowStackView.bottomAnchor.constraint(equalTo: button.bottomAnchor, constant: -12),
    ])

    return button
  }

  private func open(_ url: URL) {
    let safariViewController = SFSafariViewController(url: url)
    safariViewController.preferredControlTintColor = .systemBlue
    present(safariViewController, animated: true)
  }
}
