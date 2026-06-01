// Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?

  func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    UIApplication.shared.isIdleTimerDisabled = true
    ExternalDisplayManager.registerDefaults()
    return true
  }

  func application(
    _ application: UIApplication,
    configurationForConnecting connectingSceneSession: UISceneSession,
    options: UIScene.ConnectionOptions
  ) -> UISceneConfiguration {
    if connectingSceneSession.role.isExternalDisplay {
      guard ExternalDisplayManager.isDedicatedModeEnabled else {
        return UISceneConfiguration(
          name: nil,
          sessionRole: connectingSceneSession.role
        )
      }

      let configuration = UISceneConfiguration(
        name: "External Display Configuration",
        sessionRole: connectingSceneSession.role
      )
      configuration.delegateClass = ExternalSceneDelegate.self
      return configuration
    }

    return UISceneConfiguration(
      name: "Default Configuration",
      sessionRole: connectingSceneSession.role
    )
  }
}
