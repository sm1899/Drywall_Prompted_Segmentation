import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2


def parse_yolo_polygon(label_file):
    polygons = []
    if not os.path.exists(label_file):
        return polygons
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            
            polygon = []
            for i in range(0, len(coords), 2):
                polygon.append((coords[i], coords[i + 1]))
            polygons.append(polygon)
    
    return polygons


def parse_yolo_bbox(label_file):
    bboxes = []
    if not os.path.exists(label_file):
        return bboxes
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            bboxes.append((x_center, y_center, width, height))
    
    return bboxes


def polygon_to_mask(polygon, img_width, img_height):
    pixel_coords = []
    for x_norm, y_norm in polygon:
        x_pixel = int(x_norm * img_width)
        y_pixel = int(y_norm * img_height)
        pixel_coords.append((x_pixel, y_pixel))
    
    mask_img = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask_img)
    
    if len(pixel_coords) >= 3:
        draw.polygon(pixel_coords, fill=255)
    
    mask = np.array(mask_img, dtype=np.uint8)
    return mask


def bbox_to_mask_with_cv(bbox, img_array, img_width, img_height):
    x_center_norm, y_center_norm, width_norm, height_norm = bbox
    
    x_center = int(x_center_norm * img_width)
    y_center = int(y_center_norm * img_height)
    w = int(width_norm * img_width)
    h = int(height_norm * img_height)
    
    x1 = max(0, x_center - w // 2)
    y1 = max(0, y_center - h // 2)
    x2 = min(img_width, x_center + w // 2)
    y2 = min(img_height, y_center + h // 2)
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros((img_height, img_width), dtype=np.uint8)
    
    roi = img_array[y1:y2, x1:x2]
    
    if roi.size == 0:
        mask_roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    else:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
        
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask_roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask_roi, [largest_contour], 255)
        
        if np.sum(mask_roi) == 0:
            mask_roi = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255
    
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = mask_roi
    
    return mask


def process_dataset_split(images_dir, labels_dir, use_bbox=False):
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    label_mask_dir = labels_path.parent / 'label_mask'
    label_mask_dir.mkdir(exist_ok=True)
    
    image_files = sorted(images_path.glob('*.jpg')) + sorted(images_path.glob('*.png'))
    
    for img_file in tqdm(image_files):
        label_file = labels_path / (img_file.stem + '.txt')
        if not label_file.exists():
            continue
        
        img = Image.open(img_file)
        img_width, img_height = img.size
        img_array = np.array(img)
        
        if use_bbox:
            bboxes = parse_yolo_bbox(label_file)
            if not bboxes:
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
            else:
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                for bbox in bboxes:
                    bbox_mask = bbox_to_mask_with_cv(bbox, img_array, img_width, img_height)
                    mask = np.maximum(mask, bbox_mask)
        else:
            polygons = parse_yolo_polygon(label_file)
            if not polygons:
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
            else:
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                for polygon in polygons:
                    polygon_mask = polygon_to_mask(polygon, img_width, img_height)
                    mask = np.maximum(mask, polygon_mask)
        
        mask_filename = img_file.stem + '.png'
        mask_path = label_mask_dir / mask_filename
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(mask_path)


def main():
    base_data_dir = Path('data')
    
    cracks_dataset = base_data_dir / 'cracks.v1i.yolov5pytorch'
    if cracks_dataset.exists():
        for split in ['train', 'valid', 'test']:
            images_dir = cracks_dataset / split / 'images'
            labels_dir = cracks_dataset / split / 'labels'
            if images_dir.exists() and labels_dir.exists():
                process_dataset_split(images_dir, labels_dir, use_bbox=False)
    
    drywall_dataset = base_data_dir / 'Drywall-Join-Detect.v2i.yolov5pytorch'
    if drywall_dataset.exists():
        for split in ['train', 'valid']:
            images_dir = drywall_dataset / split / 'images'
            labels_dir = drywall_dataset / split / 'labels'
            if images_dir.exists() and labels_dir.exists():
                process_dataset_split(images_dir, labels_dir, use_bbox=True)


if __name__ == '__main__':
    main()

