# YOLO Package Examples

Here is an example of how to use YOLOPackage in an xcode project.
You will understand easy ways to use YOLO CoreML in your app for detect, segment, classify, pose, obb.

## Examples

There are currently four sample apps.

### YOLO-Single-Image-SwiftUI

A simple SwiftUI example that uses a YOLO CoreML model to infer an image selected from a photo library.

### YOLO-Single-Image-UIKit

A simple SwiftUI example that uses a YOLO CoreML model to infer an image selected from a photo library.

### YOLO-RealTime-SwiftUI

A simple sample app for real-time camera inference using SwiftUI

### YOLO-RealTime-UIKit

A simple sample app for real-time camera inference using UIKit

## Usage

1. Clone this repository.

2. Open the xcode project of the sample app you want to use.

3. Drag and drop the YOLO CoreML model file you want to use into your xcode bundle.

4. Select your account from Signing & Capabilities.

5.Build the app on a real device by clicking the Run button in Xcode.

**Note:**

The real-time inference app cannot be run on a simulator because it uses a camera.

The sample app uses local packages, so if you open multiple sample apps at the same time, the app may not be able to find the local packages. Open them one at a time.
