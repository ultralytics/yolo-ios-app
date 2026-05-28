// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

@objc(ExternalSceneDelegate)
class ExternalSceneDelegate: UIResponder, UIWindowSceneDelegate {
  var window: UIWindow?

  func scene(
    _ scene: UIScene, willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard ExternalDisplayManager.isDedicatedModeEnabled else {
      NotificationCenter.default.post(name: .externalDisplayDisconnected, object: nil)
      UIApplication.shared.requestSceneSessionDestruction(
        session, options: nil, errorHandler: nil)
      return
    }

    guard let windowScene = scene as? UIWindowScene else { return }

    let externalScreen = windowScene.screen

    // Pick the highest-resolution mode the external display supports.
    if let bestMode = externalScreen.availableModes.max(by: {
      $0.size.width * $0.size.height < $1.size.width * $1.size.height
    }) {
      externalScreen.currentMode = bestMode
    }

    externalScreen.overscanCompensation = .scale

    // Create the window and root view controller for the external scene.
    window = UIWindow(windowScene: windowScene)
    window?.frame = externalScreen.bounds

    let externalVC = ExternalViewController()
    externalVC.modalPresentationStyle = .fullScreen
    externalVC.additionalSafeAreaInsets = .zero

    window?.rootViewController = externalVC
    window?.insetsLayoutMarginsFromSafeArea = false
    window?.makeKeyAndVisible()

    // Notify the main app that the external display is up.
    NotificationCenter.default.post(
      name: .externalDisplayConnected,
      object: nil,
      userInfo: ["screen": externalScreen]
    )
  }

  func sceneDidDisconnect(_ scene: UIScene) {
    // Notify the main app that the external display has gone away.
    NotificationCenter.default.post(name: .externalDisplayDisconnected, object: nil)
    window = nil
  }
}
