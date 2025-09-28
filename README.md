# YOLO Image Segmentation Prediction Tool (YOLO Image Slicer)

This is an image recognition assistant tool based on Python and `ultralytics` YOLOv11. Its main function is to slice high-resolution images into multiple sub-tiles for separate predictions and then merge the results. This approach aims to address YOLO's weakness in detecting small objects.

## 🚀 Features

- **Full Image Prediction**: Performs a baseline prediction on the original image.
- **Multi-scale Grid Slicing**: Slices the image into 2x2, 3x3, and 4x4 grids for more detailed detection.
- **Automatic Result Merging**: Automatically converts and merges all sub-tile detection coordinates back onto the original image.
- **Visualized Output**: Generates a final result image with all detection boxes marked.

## ⚙️ Requirements

Please make sure the following Python packages are installed in your environment:

- `torch` & `torchvision`
- `ultralytics`
- `opencv-python`
- `numpy`

You can install them using pip:
```bash
pip install ultralytics opencv-python torch torchvision numpy
```

## 🛠️ How to Use

1.  **Download the Project**:
    ```bash
    git clone https://github.com/C111154350/YOLO-Image-Slicer.git
    cd YOLO-Image-Slicer
    ```
2.  **Prepare Model and Images**:
    - Place your trained YOLO model (e.g., `yolo11n.pt`) in the project folder.
    - In the `slicer.py` code, modify the `model` path to point to your model file.
    - Put the images you want to predict into a folder (e.g., `images`), and modify the `input_folder` and `output_folder` paths in the code accordingly.

3.  **Run the Script**:
    ```bash
    python slicer.py
    ```
4.  **View Results**:
    - After processing, the annotated result images will be saved in your specified `output_folder`.

## 📄 License

This project is licensed under the MIT License.