import cv2
import os
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo11n.pt")

input_folder = ""
output_folder = ""
os.makedirs(output_folder, exist_ok=True)

def predict_and_get_boxes(image, conf=0.5, iou=0.1):
   
    results = model(image, conf=conf, iou=iou)[0]
    boxes = []
    for r in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, r[:4])
        boxes.append((x1, y1, x2, y2))
    return boxes

def split_image(image, grid_size):
    
    h, w, _ = image.shape
    sub_h, sub_w = h // grid_size[0], w // grid_size[1]
    sub_images = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x_start, y_start = j * sub_w, i * sub_h
            x_end, y_end = x_start + sub_w, y_start + sub_h
            sub_images.append(image[y_start:y_end, x_start:x_end])
    return sub_images, (sub_h, sub_w)

def process_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            
            h, w, _ = image.shape
            final_image = image.copy()
            all_boxes = []

            
            boxes = predict_and_get_boxes(image)
            all_boxes.extend(boxes)

            
            for grid in [(2, 2), (3, 3), (4, 4)]:
                sub_images, (sub_h, sub_w) = split_image(image, grid)
                for idx, sub_img in enumerate(sub_images):
                    i, j = divmod(idx, grid[1])
                    offset_x, offset_y = j * sub_w, i * sub_h
                    boxes = predict_and_get_boxes(sub_img)
                    boxes_global = [((x1 + offset_x), (y1 + offset_y), (x2 + offset_x), (y2 + offset_y)) for (x1, y1, x2, y2) in boxes]
                    all_boxes.extend(boxes_global)
            
            
            for (x1, y1, x2, y2) in all_boxes:
                cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            output_path = os.path.join(output_folder, f"{filename}_all_boxes.jpg")
            cv2.imwrite(output_path, final_image)
            
process_images(input_folder, output_folder)