// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import SwiftUI

/// UIViewRepresentable wrapper for AVCaptureVideoPreviewLayer.
/// This is the only UIKit bridge in YOLOUI -- required because SwiftUI has no native camera preview.
public struct CameraPreview: UIViewRepresentable {
  let session: AVCaptureSession

  public init(session: AVCaptureSession) {
    self.session = session
  }

  public func makeUIView(context: Context) -> PreviewView {
    let view = PreviewView()
    view.previewLayer.session = session
    view.previewLayer.videoGravity = .resizeAspectFill
    return view
  }

  public func updateUIView(_ uiView: PreviewView, context: Context) {}

  public final class PreviewView: UIView {
    override public class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
  }
}
