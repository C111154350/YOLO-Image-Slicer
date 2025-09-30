import cv2
import os
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm # 匯入tqdm來顯示進度條

# =============================================================================
# 專案設定中心 (Config Center)
# =============================================================================
class Config:
    # --- 模型與路徑設定 ---
    MODEL_PATH = 'yolo11n.pt'          # 您YOLO模型的路徑
    INPUT_VIDEO_PATH = "C:\\Users\\bruce\\Videos\\螢幕錄製內容\\螢幕錄製 2025-09-30 131545.mp4"      # 您要處理的輸入影片路徑
    OUTPUT_VIDEO_PATH = "C:\\Users\\bruce\\OneDrive\\桌面\\ME\\專題影片\\特性.mp4"    # 處理完畢後要儲存的影片路徑
    CLASS_ID = 2

    # --- 預測參數 ---
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.1

    # --- 繪圖樣式 ---
    BOX_COLOR = (0, 255, 0) # 綠色
    BOX_THICKNESS = 2

# =============================================================================
# 核心函式 (無需修改)
# =============================================================================

def predict_and_get_boxes(model, image, conf, iou):
    """執行 YOLO 預測，回傳所有框的位置 (x1, y1, x2, y2)"""
    results = model(image, conf=conf, iou=iou, verbose=False)[0]
    boxes = []
    for r in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, r[:4])
        boxes.append((x1, y1, x2, y2))
    return boxes

def split_image(image, grid_size):
    """將圖片切割成 grid_size (rows, cols)"""
    h, w, _ = image.shape
    sub_h, sub_w = h // grid_size[0], w // grid_size[1]
    sub_images = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x_start, y_start = j * sub_w, i * sub_h
            x_end, y_end = x_start + sub_w, y_start + sub_h
            sub_images.append(image[y_start:y_end, x_start:x_end])
    return sub_images, (sub_h, sub_w)

# =============================================================================
# 主處理流程
# =============================================================================

def process_video(config):
    # --- 1. 初始化模型與影片 ---
    print(f"正在載入模型: {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)
    model.to('cuda')

    print(f"正在開啟影片: {config.INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("錯誤：無法開啟影片檔案。")
        return

    # --- 2. 設定影片寫入器 ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    
    print("影片處理中，請稍候...")

    # --- 3. 逐幀處理迴圈 ---
    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = frame.copy()
            all_boxes = []

            # --- 核心分割預測邏輯 (與您原版本相同) ---
            # 1. 原圖預測
            boxes = predict_and_get_boxes(model, frame, config.CONF_THRESHOLD, config.IOU_THRESHOLD)
            all_boxes.extend(boxes)

            # 2. 分割 & 偵測
            for grid in [(2, 2), (3, 3), (4, 4)]:
                sub_images, (sub_h, sub_w) = split_image(frame, grid)
                for idx, sub_img in enumerate(sub_images):
                    i, j = divmod(idx, grid[1])
                    offset_x, offset_y = j * sub_w, i * sub_h
                    
                    sub_boxes = predict_and_get_boxes(model, sub_img, config.CONF_THRESHOLD, config.IOU_THRESHOLD)
                    
                    boxes_global = [((x1 + offset_x), (y1 + offset_y), (x2 + offset_x), (y2 + offset_y)) for (x1, y1, x2, y2) in sub_boxes]
                    all_boxes.extend(boxes_global)
            
            # --- 繪製所有偵測到的框 ---
            for (x1, y1, x2, y2) in all_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS)
            
            # --- 將處理完的幀寫入新影片 ---
            out.write(annotated_frame)
            
            # (可選) 即時顯示處理畫面，按'q'可提早結束
            # cv2.imshow("Processing...", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            pbar.update(1) # 更新進度條

    # --- 4. 釋放資源 ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n處理完成！影片已儲存至: {config.OUTPUT_VIDEO_PATH}")

# =============================================================================
# 執行
# =============================================================================
if __name__ == "__main__":
    # 在執行前，請先安裝 tqdm: pip install tqdm
    process_video(Config())