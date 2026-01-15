#!/usr/bin/env python3
"""
YOLO Object Detection Script
This script uses YOLO11 for object detection on images or video files.
"""

import torch
from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--model', default='yolo11n.pt', help='Path to YOLO model')
    parser.add_argument('--source', default='images/', help='Path to image/video or directory')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save detection results')
    
    args = parser.parse_args()
    
    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {args.device}")
    
    # Load model
    model = YOLO(args.model)
    
    # Run prediction
    results = model.predict(
        source=args.source,
        device=args.device,
        conf=args.conf,
        save=args.save
    )
    
    # Print results summary
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Image: {result.path}")
            print(f"Detections: {len(boxes)}")
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = model.names[cls_id]
                print(f"  - {class_name}: {conf:.2f}")
        else:
            print(f"Image: {result.path} - No detections")

if __name__ == "__main__":
    main()