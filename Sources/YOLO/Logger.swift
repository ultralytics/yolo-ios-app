// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import OSLog

/// Unified logger for the Ultralytics YOLO SDK.
///
/// Routes diagnostic output through Apple's unified logging system so messages appear in
/// Console.app and can be filtered by subsystem. Debug builds also echo to stdout.
enum YOLOLog {
  private static let logger = Logger(subsystem: "com.ultralytics.yolo", category: "YOLO")

  static func error(_ message: @autoclosure () -> String) {
    let text = message()
    logger.error("\(text, privacy: .public)")
    #if DEBUG
      print("[YOLO] \(text)")
    #endif
  }

  static func warning(_ message: @autoclosure () -> String) {
    let text = message()
    logger.warning("\(text, privacy: .public)")
    #if DEBUG
      print("[YOLO] \(text)")
    #endif
  }

  static func info(_ message: @autoclosure () -> String) {
    let text = message()
    logger.info("\(text, privacy: .public)")
    #if DEBUG
      print("[YOLO] \(text)")
    #endif
  }
}
