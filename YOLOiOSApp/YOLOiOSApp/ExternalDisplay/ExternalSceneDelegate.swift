// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

class ExternalSceneDelegate: UIResponder, UIWindowSceneDelegate {
  var window: UIWindow?

  func scene(
    _ scene: UIScene, willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = scene as? UIWindowScene else { return }

    window = UIWindow(windowScene: windowScene)
    window?.rootViewController = ExternalViewController()
    window?.makeKeyAndVisible()

    // Notify the main app that the external display is up.
    NotificationCenter.default.post(name: .externalDisplayConnected, object: nil)
  }

  func sceneDidDisconnect(_ scene: UIScene) {
    // Notify the main app that the external display has gone away.
    NotificationCenter.default.post(name: .externalDisplayDisconnected, object: nil)
    window = nil
  }
}
