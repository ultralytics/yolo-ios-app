// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO SDK, providing the Apple Core AI (`.aimodel`) inference backend.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  Core AI runs `.aimodel` models on iOS 27 and later; Core ML remains the backend for earlier iOS versions and for the
//  iOS Simulator, which does not ship Core AI. A Core AI model has no Vision integration, so this backend letterboxes the
//  image itself and hands the raw output tensors to the same task decoders the Core ML path uses.

import Accelerate
import CoreImage
import CoreML
import Vision

#if canImport(CoreAI)
  import CoreAI
#endif

/// Vision-shaped carrier for a Core AI model, so every task predictor drives both backends through one `VNRequest`
/// and reads the outputs through `BasePredictor.featureArrays`.
final class CoreAIRequest: VNRequest, @unchecked Sendable {
  /// Output tensors of the last inference as Float32 copies, in the model's declared output order.
  var outputs: [MLMultiArray] = []

  /// Runs the model on an RGB CHW tensor normalized to 0-1.
  let infer: ([Float]) throws -> [MLMultiArray]

  /// BGRA render target and planar scratch reused across frames, sized to the model input.
  let pixelBuffer: CVPixelBuffer
  var planes: [UInt8]

  init(width: Int, height: Int, infer: @escaping ([Float]) throws -> [MLMultiArray]) throws {
    var buffer: CVPixelBuffer?
    CVPixelBufferCreate(
      nil, width, height, kCVPixelFormatType_32BGRA,
      [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
    guard let buffer else {
      throw PredictorError.invalidCoreAIModel("cannot allocate the input buffer")
    }
    self.pixelBuffer = buffer
    self.planes = [UInt8](repeating: 0, count: width * height * 4)
    self.infer = infer
    super.init(completionHandler: nil)
  }
}

extension BasePredictor {
  /// Whether this build and OS can run Core AI (`.aimodel`) models: iOS 27 or later on a device.
  public static var isCoreAIAvailable: Bool {
    #if canImport(CoreAI)
      if #available(iOS 27.0, *) { return true }
    #endif
    return false
  }

  /// Reads the Ultralytics export metadata of an `.aimodel`. It is plain JSON holding the same keys and string values
  /// as a Core ML model's creator-defined metadata, so no Core AI import is needed.
  static func coreAIMetadata(at url: URL) throws -> [String: String] {
    let data = try Data(contentsOf: url.appendingPathComponent("metadata.json"))
    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    let metadata = json?["creatorDefinedMetadata"] as? [String: Any] ?? [:]
    return metadata.mapValues { "\($0)" }
  }

  /// Loads a Core AI model and installs it as this predictor's request.
  func loadCoreAIModel(at url: URL, useGpu: Bool) throws {
    #if canImport(CoreAI)
      if #available(iOS 27.0, *) {
        let metadata = try Self.coreAIMetadata(at: url)
        labels = Self.parseLabels(from: metadata)
        requiresNMS = metadata["end2end"]?.lowercased() != "true"
        let request = try CoreAIRequest.load(url: url, useGpu: useGpu)
        modelInputSize = (
          CVPixelBufferGetWidth(request.pixelBuffer), CVPixelBufferGetHeight(request.pixelBuffer)
        )
        visionRequest = request
        return
      }
    #endif
    throw PredictorError.coreAIUnavailable
  }

  /// Letterboxes (or center-crops, for classification) the image into the model input and returns it as RGB CHW 0-1,
  /// with the same centered aspect-fit geometry as Vision's `.scaleFit` so the un-letterbox math is shared.
  func coreAIInput(from image: CIImage, for request: CoreAIRequest) throws -> [Float] {
    let width = modelInputSize.width
    let height = modelInputSize.height
    let extent = image.extent
    guard width > 0, height > 0, extent.width > 0, extent.height > 0 else {
      throw PredictorError.invalidCoreAIModel("empty input")
    }
    var gain = min(CGFloat(height) / extent.height, CGFloat(width) / extent.width)
    var padX = CGFloat(0)
    var padY = CGFloat(0)
    if imageCropAndScaleOption == .centerCrop {
      gain = max(CGFloat(height) / extent.height, CGFloat(width) / extent.width)
      padX = (CGFloat(width) - extent.width * gain) / 2
      padY = (CGFloat(height) - extent.height * gain) / 2
    } else if let transform = letterboxTransform(
      inputSize: extent.size, modelInputSize: modelInputSize)
    {
      (gain, padX, padY) = transform
    }

    // Core Image is bottom-left origin while `padY` is measured from the top row of the model input.
    let bounds = CGRect(x: 0, y: 0, width: width, height: height)
    let scaled = image.transformed(
      by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
        .concatenating(CGAffineTransform(scaleX: gain, y: gain))
        .concatenating(
          CGAffineTransform(translationX: padX, y: CGFloat(height) - padY - extent.height * gain)))
    let gray = CIColor(red: 114.0 / 255, green: 114.0 / 255, blue: 114.0 / 255)  // Ultralytics LetterBox padding
    Self.ciContext.render(
      scaled.composited(over: CIImage(color: gray)), to: request.pixelBuffer, bounds: bounds,
      colorSpace: CGColorSpaceCreateDeviceRGB())

    CVPixelBufferLockBaseAddress(request.pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(request.pixelBuffer, .readOnly) }
    var source = vImage_Buffer(
      data: CVPixelBufferGetBaseAddress(request.pixelBuffer), height: vImagePixelCount(height),
      width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(request.pixelBuffer))
    let count = width * height
    var input = [Float](repeating: 0, count: count * 3)
    request.planes.withUnsafeMutableBytes { planes in
      input.withUnsafeMutableBytes { floats in
        func plane8(_ index: Int) -> vImage_Buffer {
          vImage_Buffer(
            data: planes.baseAddress! + index * count, height: vImagePixelCount(height),
            width: vImagePixelCount(width), rowBytes: width)
        }
        // The interleaved order in memory is B, G, R, A; planes 0-2 are reordered to R, G, B below.
        var (blue, green, red, alpha) = (plane8(0), plane8(1), plane8(2), plane8(3))
        vImageConvert_ARGB8888toPlanar8(
          &source, &blue, &green, &red, &alpha, vImage_Flags(kvImageNoFlags))
        for (channel, var plane) in [red, green, blue].enumerated() {
          var destination = vImage_Buffer(
            data: floats.baseAddress! + channel * count * MemoryLayout<Float>.stride,
            height: vImagePixelCount(height), width: vImagePixelCount(width),
            rowBytes: width * MemoryLayout<Float>.stride)
          vImageConvert_Planar8toPlanarF(&plane, &destination, 1, 0, vImage_Flags(kvImageNoFlags))
        }
      }
    }
    return input
  }
}

#if canImport(CoreAI)
  @available(iOS 27.0, *)
  extension CoreAIRequest {
    /// Loads and specializes an `.aimodel`. `useGpu` keeps its SDK meaning of hardware acceleration: the Neural Engine
    /// is preferred when true and inference is pinned to the CPU when false.
    static func load(url: URL, useGpu: Bool) throws -> CoreAIRequest {
      let options: SpecializationOptions =
        useGpu ? SpecializationOptions(preferredComputeUnitKind: .neuralEngine) : .cpuOnly
      let function = try blocking {
        try await AIModel(contentsOf: url, options: options).loadFunction(named: "main")
      }
      guard let function, let inputName = function.descriptor.inputNames.first,
        case .ndArray(let input)? = function.descriptor.inputDescriptor(of: inputName),
        input.shape.count == 4
      else {
        throw PredictorError.invalidCoreAIModel("expected one BCHW tensor input in function 'main'")
      }
      let shape = input.shape
      let isHalf = input.scalarType == .float16
      let outputNames = function.descriptor.outputNames

      return try CoreAIRequest(width: shape[3], height: shape[2]) { floats in
        try blocking {
          var outputs = try await function.run(
            inputs: [
              inputName: isHalf
                ? NDArray(scalars: half(floats), shape: shape)
                : NDArray(scalars: floats, shape: shape)
            ])
          return try outputNames.map { name in
            guard let array = outputs.remove(name)?.ndArray else {
              throw PredictorError.invalidCoreAIModel("missing output '\(name)'")
            }
            return try multiArray(from: array)
          }
        }
      }
    }

    private static func half(_ floats: [Float]) -> [Float16] {
      var halves = [Float16](repeating: 0, count: floats.count)
      floats.withUnsafeBytes { source in
        halves.withUnsafeMutableBytes { destination in
          var src = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: source.baseAddress!), height: 1,
            width: vImagePixelCount(floats.count), rowBytes: source.count)
          var dst = vImage_Buffer(
            data: destination.baseAddress!, height: 1, width: vImagePixelCount(floats.count),
            rowBytes: destination.count)
          vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
        }
      }
      return halves
    }

    /// Copies a Core AI output into the Float32 `MLMultiArray` layout the task decoders read.
    private static func multiArray(from array: NDArray) throws -> MLMultiArray {
      let result = try MLMultiArray(
        shape: array.shape.map { NSNumber(value: $0) }, dataType: .float32)
      let count = result.count
      let destination = result.dataPointer.bindMemory(to: Float.self, capacity: count)
      func copy<T: BitwiseCopyable>(_ type: T.Type, _ body: (UnsafeBufferPointer<T>) -> Void) throws
      {
        guard let elements = array.view(as: type).contiguousElements, elements.count == count else {
          throw PredictorError.invalidCoreAIModel("non-contiguous output")
        }
        elements.withUnsafeBufferPointer(body)
      }
      switch array.scalarType {
      case .float32:
        try copy(Float.self) { destination.update(from: $0.baseAddress!, count: count) }
      case .float16:
        try copy(Float16.self) { source in
          var src = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: source.baseAddress!), height: 1,
            width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float16>.stride)
          var dst = vImage_Buffer(
            data: destination, height: 1, width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.stride)
          vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
        }
      default:
        throw PredictorError.invalidCoreAIModel("unsupported output type \(array.scalarType)")
      }
      return result
    }

    /// Runs an async Core AI call to completion. Predictors are synchronous and run off the main actor.
    private static func blocking<T>(_ operation: @escaping () async throws -> T) throws -> T {
      let result = ResultBox<T>()
      let semaphore = DispatchSemaphore(value: 0)
      Task.detached(priority: .userInitiated) {
        do { result.value = .success(try await operation()) } catch {
          result.value = .failure(error)
        }
        semaphore.signal()
      }
      semaphore.wait()
      return try result.value!.get()
    }

    private final class ResultBox<T>: @unchecked Sendable {
      var value: Result<T, Error>?
    }
  }
#endif
