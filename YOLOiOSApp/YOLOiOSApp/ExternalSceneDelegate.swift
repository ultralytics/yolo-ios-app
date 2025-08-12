// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import UIKit

@objc(ExternalSceneDelegate)
class ExternalSceneDelegate: UIResponder, UIWindowSceneDelegate {
  var window: UIWindow?

  func scene(
    _ scene: UIScene, willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = scene as? UIWindowScene else { return }

    let externalScreen = windowScene.screen

    // Select highest resolution
    if let bestMode = externalScreen.availableModes.max(by: {
      $0.size.width * $0.size.height < $1.size.width * $1.size.height
    }) {
      externalScreen.currentMode = bestMode
    }

    externalScreen.overscanCompensation = .scale

    // Setup window and controller
    window = UIWindow(windowScene: windowScene)
    window?.frame = externalScreen.bounds

    let externalVC = ExternalViewController()
    externalVC.modalPresentationStyle = .fullScreen
    externalVC.additionalSafeAreaInsets = .zero

    window?.rootViewController = externalVC
    window?.insetsLayoutMarginsFromSafeArea = false
    window?.makeKeyAndVisible()

    // Notify connection
    NotificationCenter.default.post(
      name: .externalDisplayConnected,
      object: nil,
      userInfo: ["screen": externalScreen]
    )
  }

  func sceneDidDisconnect(_ scene: UIScene) {
    // Notify main app that external display is disconnected
    NotificationCenter.default.post(name: .externalDisplayDisconnected, object: nil)
    window = nil
  }
}
