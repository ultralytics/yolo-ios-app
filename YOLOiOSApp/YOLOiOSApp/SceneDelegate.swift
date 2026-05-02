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

  @objc private func handleExternalDisplayConnected() {
    SceneDelegate.hasExternalDisplay = true
    updateOrientationLock()
  }

  @objc private func handleExternalDisplayDisconnected() {
    SceneDelegate.hasExternalDisplay = false
    updateOrientationLock()
  }

  private func updateOrientationLock() {
    guard let windowScene = window?.windowScene else { return }
    windowScene.requestGeometryUpdate(
      .iOS(interfaceOrientations: supportedInterfaceOrientations))
    window?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
  }

  var supportedInterfaceOrientations: UIInterfaceOrientationMask {
    // Only support landscape when external display is connected
    if SceneDelegate.hasExternalDisplay {
      return [.landscapeLeft, .landscapeRight]
    } else {
      // Support all orientations when no external display
      return [.portrait, .landscapeLeft, .landscapeRight]
    }
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }
}
