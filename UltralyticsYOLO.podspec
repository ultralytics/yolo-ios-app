# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

Pod::Spec.new do |s|
  s.name             = 'UltralyticsYOLO'
  s.version          = '8.9.2'
  s.summary          = 'Ultralytics YOLO core inference for iOS (CoreML/Vision).'
  s.homepage         = 'https://github.com/ultralytics/yolo-ios-app'
  s.license          = { :type => 'AGPL-3.0', :file => 'LICENSE' }
  s.author           = { 'Ultralytics' => 'info@ultralytics.com' }
  s.source           = { :git => 'https://github.com/ultralytics/yolo-ios-app.git', :tag => "v#{s.version}" }
  s.source_files     = 'Sources/UltralyticsYOLO/**/*.swift'
  s.ios.deployment_target = '13.0'
  s.swift_version    = '5.10'
end
