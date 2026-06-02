// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import OSLog

/// Unified logger for the Ultralytics YOLO SDK.
///
/// Routes diagnostic output through Apple's unified logging system so messages appear in Console.app and can be
/// filtered by subsystem. Debug builds also echo to stdout.
enum YOLOLog {
  private static let subsystem = "com.ultralytics.yolo"
  private static let category = "YOLO"
  private static let legacyLog = OSLog(subsystem: subsystem, category: category)

  static func error(_ message: @autoclosure () -> String) {
    let text = message()
    if #available(iOS 14.0, *) {
      Logger(subsystem: subsystem, category: category).error("\(text, privacy: .public)")
    } else {
      os_log("%{public}@", log: legacyLog, type: .error, text)
    }
    #if DEBUG
      print("[YOLO] \(text)")
    #endif
  }

  static func warning(_ message: @autoclosure () -> String) {
    let text = message()
    if #available(iOS 14.0, *) {
      Logger(subsystem: subsystem, category: category).warning("\(text, privacy: .public)")
    } else {
      os_log("%{public}@", log: legacyLog, type: .default, text)
    }
    #if DEBUG
      print("[YOLO] \(text)")
    #endif
  }

  static func info(_ message: @autoclosure () -> String) {
    let text = message()
    if #available(iOS 14.0, *) {
      Logger(subsystem: subsystem, category: category).info("\(text, privacy: .public)")
    } else {
      os_log("%{public}@", log: legacyLog, type: .info, text)
    }
    #if DEBUG
      print("[YOLO] \(text)")
    #endif
  }
}
