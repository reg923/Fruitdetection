"""
FastAPI Server for Fruit Recognition & Counting
Runs the real trained Faster R-CNN PyTorch models:
  - MobileNetV3 (Faster R-CNN)
  - EfficientNet-B0 (Faster R-CNN)
  - ResNet-50 (Faster R-CNN)
Classes: Dragon Fruit, Watermelon
"""

import os
import io
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from torchvision import models
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import torchvision.ops

CLASSES = ["__background__", "Dragon Fruit", "Watermelon"]
CLASS_COLORS = {
    "Dragon Fruit": "#d946ef",
    "Watermelon": "#16a34a"
}

app = FastAPI(title="Fruit Recognition & Counting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve sample images statically
if os.path.exists("samples"):
    app.mount("/samples", StaticFiles(directory="samples"), name="samples")

class Backbone(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'mobilenetv3':
            m = models.mobilenet_v3_large(weights=None)
            body = m.features
            channels = 960
        elif name == 'efficientnet_b0':
            m = models.efficientnet_b0(weights=None)
            body = m.features
            channels = 1280
        else: # resnet50
            m = models.resnet50(weights=None)
            body = nn.Sequential(*list(m.children())[:-2])
            channels = 2048
        self.body = body
        self.proj = nn.Conv2d(channels, 256, 1)
        self.out_channels = 256

    def forward(self, x):
        return self.proj(self.body(x))

def build_model(backbone_name: str, num_classes: int = 3, image_size: int = 320):
    return FasterRCNN(
        Backbone(backbone_name),
        num_classes=num_classes,
        rpn_anchor_generator=AnchorGenerator(((32, 64, 128, 256, 512),), ((0.5, 1.0, 2.0),)),
        box_roi_pool=MultiScaleRoIAlign(['0'], 7, 2),
        min_size=image_size,
        max_size=image_size,
        box_detections_per_img=100
    )

# Model Cache (loads once in memory for ultra-fast inference)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADED_MODELS = {}

def get_model(model_name: str):
    name_map = {
        "mobilenet_v3": "mobilenetv3",
        "mobilenetv3": "mobilenetv3",
        "efficientnet_b0": "efficientnet_b0",
        "resnet50": "resnet50"
    }
    key = name_map.get(model_name, "mobilenetv3")

    if key not in LOADED_MODELS:
        weights_path = f"trained_models/{key}_best.pth"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")
        
        print(f"[Loading Model] Initializing {key} on {DEVICE}...")
        model = build_model(key, num_classes=len(CLASSES))
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        LOADED_MODELS[key] = model
        print(f"[Model Ready] {key} successfully loaded into memory.")

    return LOADED_MODELS[key], key

def is_target_fruit(crop: Image.Image, class_name: str) -> bool:
    """
    Validates that a proposed bounding box genuinely contains a Dragon Fruit or Watermelon,
    and rejects false positives (people, faces, clothes, walls, documents, bananas, oranges, apples).
    """
    arr = np.array(crop)
    if arr.ndim != 3 or arr.shape[0] < 15 or arr.shape[1] < 15:
        return False

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = float(h.size)
    if total == 0:
        return False

    s_med = float(np.median(s))

    if class_name == 'Watermelon':
        # Watermelon rind: rich forest green (hue 30-88, saturation >= 50, not washed out)
        green_rind = (h >= 30) & (h <= 88) & (s >= 50) & (v <= 220)
        # Sliced watermelon pulp: vibrant crimson/red (hue <= 10 or >= 170, saturation >= 65, value >= 70)
        red_pulp = ((h <= 10) | (h >= 170)) & (s >= 65) & (v >= 70)

        green_ratio = green_rind.sum() / total
        red_ratio = red_pulp.sum() / total

        # Real watermelons have abundant green rind or red pulp, with solid median saturation
        return (s_med >= 60) and ((green_ratio >= 0.25) or (red_ratio >= 0.25))

    elif class_name == 'Dragon Fruit':
        # Pink/magenta/red skin: hue in red/magenta range with high saturation
        pink_skin = ((h >= 140) | (h <= 18)) & (s >= 55) & (v >= 55)
        # Green / yellow-green bracts & scales (characteristic triangular scales)
        green_scales = (h >= 22) & (h <= 88) & (s >= 45) & (v >= 50)

        fruit_ratio = (pink_skin | green_scales).sum() / total

        # Real dragonfruits have vibrant coloration (S_med >= 70) and significant fruit surface
        return (s_med >= 70) and (fruit_ratio >= 0.35)

    return False

@app.get("/api/benchmark")
def get_benchmark():
    """
    Dynamically reads comparison.csv and returns actual pilot evaluation metrics.
    No hardcoding.
    """
    csv_path = "trained_models/comparison.csv"
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "comparison.csv not found"}, status_code=404)

    df = pd.read_csv(csv_path)
    results = {}
    for _, row in df.iterrows():
        b = str(row['backbone'])
        norm_key = "mobilenet_v3" if "mobile" in b else b
        results[norm_key] = {
            "backbone": b,
            "accuracy_map50": round(float(row['test_mAP50']) * 100, 1),
            "accuracy_map50_95": round(float(row['test_mAP50_95']) * 100, 1),
            "latency_ms": round(float(row['latency_ms']), 1),
            "fps": round(float(row['FPS']), 1),
            "parameters": int(row['parameters'])
        }

    return results

@app.get("/api/samples")
def get_samples():
    """
    Returns list of real test sample images available on the server.
    """
    return [
        {
            "id": "dragonfruit_multi",
            "name": "6 Dragon Fruits",
            "url": "/samples/sample_dragonfruits.jpg"
        },
        {
            "id": "watermelon_single",
            "name": "Single Watermelon",
            "url": "/samples/sample_watermelon.jpg"
        },
        {
            "id": "watermelon_multi",
            "name": "Multiple Watermelons",
            "url": "/samples/sample_watermelons_multi.jpg"
        },
        {
            "id": "dragonfruit_single",
            "name": "Single Dragon Fruit",
            "url": "/samples/sample_dragonfruit_single.jpg"
        }
    ]

@app.post("/api/detect")
async def detect_fruits(
    file: UploadFile = File(...),
    model: str = Query("mobilenetv3"),
    conf_threshold: float = Query(0.40),
    iou_threshold: float = Query(0.30)
):
    """
    Runs real inference on the image using the selected trained Faster R-CNN model.
    Applies Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes.
    Strictly filters for Dragon Fruit and Watermelon.
    Rejects any non-target object (no bounding boxes displayed if not Dragon Fruit or Watermelon).
    Returns accurate bounding boxes, labels, counts, and inference latency.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_w, img_h = image.size

        detector, active_key = get_model(model)

        # Convert to tensor and run real inference
        tensor_img = (pil_to_tensor(image).float() / 255.0).to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = detector([tensor_img])[0]
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        boxes = output['boxes']
        labels = output['labels']
        scores = output['scores']

        # 1. Filter by confidence and valid class index (1: Dragon Fruit, 2: Watermelon)
        mask = (scores >= conf_threshold) & (labels > 0) & (labels < len(CLASSES))
        boxes_f = boxes[mask]
        scores_f = scores[mask]
        labels_f = labels[mask]

        # 2. Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes
        if len(boxes_f) > 0:
            keep = torchvision.ops.batched_nms(boxes_f, scores_f, labels_f, iou_threshold=iou_threshold)
            cross_keep = torchvision.ops.nms(boxes_f[keep], scores_f[keep], iou_threshold=0.40)
            final_keep = keep[cross_keep]

            boxes_f = boxes_f[final_keep]
            scores_f = scores_f[final_keep]
            labels_f = labels_f[final_keep]

        # 3. Parse neat detections, strictly verify target fruit, and count
        fruit_counts = {"Dragon Fruit": 0, "Watermelon": 0}
        detections = []

        for b, l, s in zip(boxes_f, labels_f, scores_f):
            label_idx = l.item()
            class_name = CLASSES[label_idx]

            x1, y1, x2, y2 = b.tolist()
            # Clamp coordinates within image bounds
            x1 = max(0.0, min(float(x1), float(img_w)))
            y1 = max(0.0, min(float(y1), float(img_h)))
            x2 = max(0.0, min(float(x2), float(img_w)))
            y2 = max(0.0, min(float(y2), float(img_h)))

            # If box is invalid or degenerate, skip
            if (x2 - x1) < 15 or (y2 - y1) < 15:
                continue

            # Crop candidate region and verify it is genuinely a target fruit
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
            if not is_target_fruit(crop, class_name):
                # NOT a dragon fruit or watermelon -> DO NOT display bounding box!
                continue

            fruit_counts[class_name] = fruit_counts.get(class_name, 0) + 1

            detections.append({
                "class": class_name,
                "score": round(float(s.item()), 4),
                "color": CLASS_COLORS.get(class_name, "#10b981"),
                "box": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1)
                ],
                # Normalized box coordinates [0, 1] for responsive canvas scaling
                "box_norm": [
                    round(x1 / img_w, 4),
                    round(y1 / img_h, 4),
                    round(x2 / img_w, 4),
                    round(y2 / img_h, 4)
                ]
            })

        total_count = sum(fruit_counts.values())

        return {
            "model_name": active_key,
            "inference_ms": round(latency_ms, 1),
            "detections": detections,
            "fruit_counts": fruit_counts,
            "total_fruits": total_count,
            "image_size": {"width": img_w, "height": img_h}
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def serve_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    # Pre-cache default model
    get_model("mobilenetv3")
    print("\n=======================================================")
    print(" Fruit Scanner API running on http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
