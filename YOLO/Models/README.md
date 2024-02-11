<img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320">

# Add YOLOv8 Models to the Project

To utilize the full power of the Ultralytics YOLO iOS App, you'll need to add YOLOv8 models. These models are not included directly in the repository for two main reasons:

1. **File Size:** YOLOv8 models can be large (i.e. up to 60 MB), making the repository heavy and cumbersome to clone or download.
2. **Frequent Updates:** We continuously improve and update YOLOv8 models to enhance performance and accuracy. Including them directly in the repository would make it challenging to keep the app up to date with the latest models.

There are two ways to add YOLOv8 models to your project:

## Option 1: Download from GitHub Release Assets

For convenience, we provide pre-compiled and optimized YOLOv8 models as release assets on our GitHub repository. This method ensures you get the latest, ready-to-use models without needing additional steps.

- Visit the [Ultralytics YOLO iOS App GitHub release assets page](https://github.com/ultralytics/yolo-ios-app).
- Download the desired YOLOv8 model files.
- Place the downloaded model files into the `YOLO/Models` directory of your project.

## Option 2: Export Models Using the Ultralytics Python Package

If you prefer to use specific model versions or need to customize the models, you can export them using the `ultralytics` Python package. This approach provides flexibility in selecting and optimizing models for your specific application needs.

1. **Installation:** First, ensure you have the `ultralytics` package installed. If not, you can install it using pip:

    ```sh
    pip install ultralytics
    ```

2. **Export Models:** Use the following Python script to export YOLOv8 models to the CoreML format, optimized for INT8 quantization for better performance on iOS devices. The script exports all YOLOv8 model sizes (`n`, `s`, `m`, `l`, `x`) as CoreML models.

    ```python
    from ultralytics import YOLO

    # Export all YOLOv8 models to CoreML INT8 
    for size in ("n", "s", "m", "l", "x"):  # all YOLOv8 model sizes
        YOLO(f"yolov8{size}.pt").export(format="coreml", int8=True, nms=True, imgsz=[640, 384])
    ```

3. **Place Models in Project:** After exporting, locate the CoreML model files and place them in the `YOLO/Models` directory of your project.

## Finalizing the Setup

Once you've added the models to the `YOLO/Models` directory by either downloading them from GitHub or exporting them using the Ultralytics package, your Ultralytics YOLO iOS App is ready to detect objects with high accuracy and performance.

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/ultralytics/assets/26833433/bbe30e03-65c3-4cdb-9f15-2163f3147dbc" alt="Ultralytics YOLO XCode screenshot">
</p>

By offering these two options, we aim to provide flexibility and ensure you have access to the latest advancements in object detection technology. Whether you're a developer, researcher, or enthusiast, these options allow you to integrate YOLOv8 models into your iOS projects efficiently and effectively.
