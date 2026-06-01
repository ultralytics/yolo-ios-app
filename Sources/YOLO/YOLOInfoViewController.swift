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
      subtitle: "Training, prediction, Core ML export, and deployment guides.",
      systemImage: "book",
      url: URL(string: "https://docs.ultralytics.com")!
    ),
    Resource(
      title: "YOLO Models",
      subtitle: "Explore YOLO26 models, supported tasks, sizes, and performance.",
      systemImage: "square.stack.3d.up",
      url: URL(string: "https://platform.ultralytics.com/ultralytics/yolo26")!
    ),
    Resource(
      title: "GitHub",
      subtitle: "Explore the main Ultralytics package, releases, and open-source tools.",
      systemImage: "chevron.left.forwardslash.chevron.right",
      url: URL(string: "https://github.com/ultralytics/ultralytics")!
    ),
    Resource(
      title: "Licensing",
      subtitle: "Review AGPL-3.0 and Enterprise License options for commercial use.",
      systemImage: "doc.text",
      url: URL(string: "https://www.ultralytics.com/license")!
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

    let stackView = stackView(spacing: 18)
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
    let stackView = stackView(spacing: 8)
    stackView.alignment = .center

    let iconView = UIImageView(image: UIImage(systemName: "camera.viewfinder"))
    iconView.tintColor = .systemBlue
    iconView.contentMode = .scaleAspectFit
    iconView.translatesAutoresizingMaskIntoConstraints = false

    let titleLabel = label("Ultralytics YOLO", style: .title1)
    titleLabel.textAlignment = .center

    let subtitleLabel = label("Real-time AI vision for iOS", style: .subheadline)
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
    let stackView = stackView(spacing: 6)
    let titleLabel = label(title, style: .headline)
    let bodyLabel = label(body, style: .body)
    bodyLabel.textColor = .secondaryLabel
    bodyLabel.numberOfLines = 0

    stackView.addArrangedSubview(titleLabel)
    stackView.addArrangedSubview(bodyLabel)
    return stackView
  }

  private func resourcesSection() -> UIView {
    let stackView = stackView(spacing: 10)

    let titleLabel = label("Continue Learning", style: .headline)
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

    let titleLabel = label(resource.title, style: .headline)
    titleLabel.textColor = .systemBlue

    let subtitleLabel = label(resource.subtitle, style: .footnote)
    subtitleLabel.textColor = .secondaryLabel
    subtitleLabel.numberOfLines = 0

    let textStackView = stackView([titleLabel, subtitleLabel], spacing: 2)

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

  private func label(_ text: String, style: UIFont.TextStyle) -> UILabel {
    let label = UILabel()
    label.text = text
    label.font = .preferredFont(forTextStyle: style)
    label.adjustsFontForContentSizeCategory = true
    return label
  }

  private func stackView(_ views: [UIView] = [], spacing: CGFloat) -> UIStackView {
    let stackView = UIStackView(arrangedSubviews: views)
    stackView.axis = .vertical
    stackView.spacing = spacing
    return stackView
  }

  private func open(_ url: URL) {
    let safariViewController = SFSafariViewController(url: url)
    safariViewController.preferredControlTintColor = .systemBlue
    present(safariViewController, animated: true)
  }
}
