// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
// Safe reading of MLMultiArray without assuming Float32. Core ML models may output
// Float32 or Double; assumingMemoryBound(to: Float.self) on Double data misinterprets
// memory and produces wrong results or crashes.

import CoreML

extension MLMultiArray {

  /// Returns a function that reads the element at the given linear index as `Float`.
  /// Handles both `.float32` and `.double` storage; other types use subscript (slower).
  func makeFloatReader() -> (Int) -> Float {
    switch dataType {
    case .float32:
      let ptr = dataPointer.assumingMemoryBound(to: Float.self)
      return { ptr[$0] }
    case .double:
      let ptr = dataPointer.assumingMemoryBound(to: Double.self)
      return { Float(ptr[$0]) }
    default:
      return { [self] index in
        self[[index] as [NSNumber]].floatValue
      }
    }
  }
}
