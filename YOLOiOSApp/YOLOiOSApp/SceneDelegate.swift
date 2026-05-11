// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {

  var window: UIWindow?

  // Track external display connection state (thread-safe)
  private static var _hasExternalDisplay = false
  private static let externalDisplayQueue = DispatchQueue(
    label: "com.ultralytics.externalDisplay.state")

  static var hasExternalDisplay: Bool {
    get {
      return externalDisplayQueue.sync { _hasExternalDisplay }
    }
    set {
      externalDisplayQueue.sync { _hasExternalDisplay = newValue }
    }
  }

  func scene(
    _ scene: UIScene, willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = (scene as? UIWindowScene) else { return }

    // Create window with the scene
    window = UIWindow(windowScene: windowScene)

    // Load the main storyboard and instantiate the initial view controller
    let storyboard = UIStoryboard(name: "Main", bundle: nil)
    if let initialVC = storyboard.instantiateInitialViewController() {
      window?.rootViewController = initialVC
    }

    window?.makeKeyAndVisible()

    // Listen for external display connection/disconnection
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayConnected),
      name: .externalDisplayConnected,
      object: nil
    )

    NotificationCenter.default.addObserver(
      self,
      selector: #selector(handleExternalDisplayDisconnected),
      name: .externalDisplayDisconnected,
      object: nil
    )
  }

  func sceneDidBecomeActive(_ scene: UIScene) {
    ExternalDisplayManager.refreshModeIfNeeded()
  }

  @objc private func handleExternalDisplayConnected() {
    SceneDelegate.hasExternalDisplay = true
  }

  @objc private func handleExternalDisplayDisconnected() {
    SceneDelegate.hasExternalDisplay = false
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }
}
