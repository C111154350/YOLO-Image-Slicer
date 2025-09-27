# YOLO 影像分割預測工具 (YOLO Image Slicer)

這是一個基於 Python 和 `ultralytics` YOLOv11 的影像辨識輔助工具。主要功能是透過將高解析度圖片切割成多個子圖塊進行分開預測，再將結果合併，旨在解決 YOLO 對於小尺寸物件偵測能力較弱的問題。

## 🚀 功能特色 (Features)

- **全圖預測**：首先對原始圖片進行一次基準預測。
- **多尺度網格分割**：將圖片切割成 2x2, 3x3, 4x4 的網格進行更細緻的偵測。
- **結果自動合併**：自動將所有子圖塊的偵測結果座標轉換並合併到原始圖片上。
- **視覺化輸出**：產生一張標示了所有偵測框的最終結果圖。

## ⚙️ 環境需求 (Requirements)

請確保您的 Python 環境中已安裝以下套件：

- `torch` & `torchvision`
- `ultralytics`
- `opencv-python`
- `numpy`

您可以使用 pip 來安裝：
```bash
pip install ultralytics opencv-python torch torchvision numpy
```

## 🛠️ 如何使用 (How to Use)

1.  **下載專案**：
    ```bash
    git clone [https://github.com/C111154350/YOLO-Image-Slicer.git](https://github.com/C111154350/YOLO-Image-Slicer.git)
    cd YOLO-Image-Slicer
    ```
2.  **準備模型與圖片**：
    - 將您訓練好的 YOLO 模型 (例如 `yolo11n.pt`) 放入專案資料夾。
    - 在程式碼 `slicer.py` 中，修改 `model` 的路徑，使其指向您的模型檔案。
    - 將您要進行預測的圖片放入一個資料夾中（例如 `images`），並在程式碼中修改 `input_folder` 和 `output_folder` 的路徑。

3.  **執行腳本**：
    ```bash
    python slicer.py
    ```
4.  **查看結果**：
    - 處理完成後，標註好的結果圖片將會儲存在您指定的 `output_folder` 中。

## 📄 授權 (License)

本專案採用 MIT 授權。