# YOLO Package Examples

Here are examples of how to use YOLO Package in an Xcode project that demonstrate how to use YOLO CoreML in your app for Detect, Segment, Classify, Pose, and Obb tasks.

## Examples

There are currently four sample apps.

### YOLO-Single-Image-SwiftUI

A simple SwiftUI example that uses a YOLO CoreML model to infer an image selected from a photo library.

### YOLO-Single-Image-UIKit

A simple UIKit example that uses a YOLO CoreML model to infer an image selected from a photo library.

### YOLO-RealTime-SwiftUI

A simple sample app for real-time camera inference using SwiftUI.

### YOLO-RealTime-UIKit

A simple sample app for real-time camera inference using UIKit.

## Usage

1. Clone this repository.

2. Open the Xcode project of the sample app you want to use.

3. Drag and drop the YOLO CoreML model file you want to use into your Xcode bundle.

4. Select your account from Signing & Capabilities.

5. Build the app on a real device by clicking the Run button in Xcode.

**Note:**

The real-time inference app cannot be run on a simulator because it uses a camera.

The sample app uses local packages, so if you open multiple sample apps at the same time, the app may not be able to find the local packages. Open them one at a time.

## How to Obtain YOLO CoreML Models

You can obtain YOLO CoreML models using either of the following methods:

### Download from GitHub Release Assets

You can download the CoreML INT8 models directly from the official YOLO GitHub release page.

Download YOLO CoreML Models (GitHub)

Place the downloaded models into your Xcode project directory.

### Export using Python

You can also export CoreML INT8 models yourself using the ultralytics Python package.

First, install the required package:

pip install ultralytics

Then, run the following Python script to export the desired models:

```
from ultralytics import YOLO
# Export for all YOLO11 model sizes
for size in ("n", "s", "m", "l", "x"):
    # Load a YOLO11 PyTorch model
    model = YOLO(f"yolo11{size}.pt")

    # Export the PyTorch model to CoreML INT8 format (with NMS layers)
    model.export(format="coreml", int8=True, nms=True, imgsz=[640, 384])

    # You can specify different task models as follows:
    # model = YOLO(f"yolo11{size}-seg.pt")   # segmentation
    # model = YOLO(f"yolo11{size}-cls.pt")   # classification
    # model = YOLO(f"yolo11{size}-pose.pt")  # pose estimation
    # model = YOLO(f"yolo11{size}-obb.pt")   # oriented bounding box

    # Export the PyTorch model to CoreML INT8 format (without NMS layers)
    model.export(format="coreml", int8=True, imgsz=[640, 384]) # For use with the package, do not add NMS to any models other than detection.
```

## Testing

Each example app includes a suite of unit tests that verify its functionality. The tests are designed to run both with and without model files to support different development and CI scenarios.

### Running Tests

To run the tests:

1. Open the example app project in Xcode
2. Select the test target (e.g., YOLO-Single-Image-SwiftUITests)
3. Press Cmd+U or select Product > Test from the menu

### Test Configuration

By default, tests are configured to run without requiring model files by setting `SKIP_MODEL_TESTS = true`. This allows testing basic functionality without needing the large CoreML models.

If you want to test with models:

1. Add the appropriate CoreML models to the main application target (see each test's README for specific model requirements)
2. Set `SKIP_MODEL_TESTS = false` in the test file
3. Run the tests again

See the README.md in each test directory for detailed instructions on:

- Which model files are required for full testing
- How to obtain and convert models
- How to properly add models to the project
- What functionality is being tested
