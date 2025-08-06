## Summary

- Fixes all 25 build warnings when building with Xcode 16.4 for iOS 18.6
- Addresses issue #127

## Changes

### Swift 6 Concurrency Warnings

- Added `@unchecked Sendable` conformance to `Classifier` and `ObjectDetector` classes
- Fixed actor isolation warning in `YOLOView` by making `photoOutput` method `nonisolated` and wrapping main actor code in `Task`

### Unused Variable Warnings

- Fixed unused variable `recognitions` in `Classifier.swift` (line 136)
- Fixed unused variable `speed` in `ObjectDetector.swift` (line 151)
- Fixed unused variable `resultsQueue` in `Segmenter.swift` (line 246)
- Fixed multiple unused `labelRect` warnings in `Plot.swift` by changing `var` to `let` where appropriate

### Deprecated API Warning

- Fixed deprecated `isHighResolutionCaptureEnabled` API in `VideoCapture.swift` by using `maxPhotoDimensions` for iOS 16.0+

## Testing

- Built package with `swift build` - no warnings
- Verified all changes maintain existing functionality
- Tested on Xcode 16.4 with iOS 18.6 deployment target

## Related Issues

- Fixes #127
  EOF < /dev/null
