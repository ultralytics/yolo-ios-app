# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

Pod::Spec.new do |s|
  s.name             = 'UltralyticsYOLO'
  s.version          = '9.0.0'
  s.summary          = 'Ultralytics YOLO core inference for iOS (CoreML/Vision).'
  s.homepage         = 'https://github.com/ultralytics/yolo-ios-app'
  s.license          = { :type => 'AGPL-3.0', :file => 'LICENSE' }
  s.author           = { 'Ultralytics' => 'info@ultralytics.com' }
  s.source           = { :git => 'https://github.com/ultralytics/yolo-ios-app.git', :tag => "v#{s.version}" }
  s.source_files     = 'Sources/UltralyticsYOLO/**/*.swift'
  s.ios.deployment_target = '13.0'
  # Inference postprocessing sweeps millions of pixels per frame; without optimization a consumer's
  # Debug build makes the SDK look ~100x slower than it ships. Always compile the pod optimized.
  s.pod_target_xcconfig = { 'SWIFT_OPTIMIZATION_LEVEL' => '-O' }
  s.swift_version    = '5.10'
end
