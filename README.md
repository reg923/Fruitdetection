# Lightweight On-Device Fruit Recognition (Edge AI)

An applied Edge AI research and benchmarking project comparing **MobileNetV3**, **ResNet-50**, and **EfficientNet-B0** for real-time bounding-box fruit detection on low-power hardware.

---

## 📁 Project Structure

```text
fruit-recognition-edge-ai/
├── index.html            # Interactive showcase website (Live detector playground & benchmark dashboard)
├── train_detector.py     # PyTorch detection training script (MobileNetV3, ResNet50, EfficientNet-B0)
├── dataset_loader.py     # Dataset loader supporting YOLO (.txt) and Pascal VOC (.xml) bounding boxes
├── benchmark_models.py   # Latency (ms), Throughput (FPS), Model size (MB) & Parameter profiler
├── export_onnx.py        # PyTorch to ONNX converter for edge / web deployment
└── fruit_dataset/        # (Your dataset folder)
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

---

## 🍎 1. Dataset Setup (Bounding Boxes)

Organize your annotated images into `fruit_dataset/`:

- **Images**: Put `.jpg` or `.png` images in `fruit_dataset/images/train/` and `fruit_dataset/images/val/`.
- **Labels**: Put the corresponding `.txt` (YOLO format) or `.xml` (Pascal VOC) files with the same filename in `fruit_dataset/labels/train/` and `fruit_dataset/labels/val/`.

### YOLO Label Format (Recommended):
```text
<class_id> <x_center> <y_center> <width> <height>
```
*(Normalized between 0.0 and 1.0)*

Example for Apple (`0`) and Banana (`1`):
```text
0 0.452 0.320 0.280 0.410
1 0.780 0.550 0.340 0.480
```

---

## 🚀 2. Training the 3 Models

### Step 1: Train MobileNetV3-Large (SSDLite) - Lightweight Edge Champion
```bash
python train_detector.py --model mobilenet_v3 --epochs 25 --batch_size 16 --lr 0.0005
```

### Step 2: Train EfficientNet-B0 - Balanced Accuracy & Speed
```bash
python train_detector.py --model efficientnet_b0 --epochs 25 --batch_size 8 --lr 0.0005
```

### Step 3: Train ResNet-50 v2 (Faster R-CNN) - High Precision Baseline
```bash
python train_detector.py --model resnet50 --epochs 25 --batch_size 4 --lr 0.0002
```

Checkpoints will automatically save to `./checkpoints/<model_name>_best.pth`.

---

## 📊 3. Benchmarking Latency, FPS, and Size

Run the benchmark profiler across all 3 models:
```bash
python benchmark_models.py --model all
```

Output:
```text
================== EXECUTIVE SUMMARY MATRIX ==================
Model              | Size (MB)  | Params (M) | Latency (ms) | FPS     
--------------------------------------------------------------------
mobilenet_v3       | 13.20      | 3.42       | 16.40        | 61.0    
efficientnet_b0    | 26.80      | 7.85       | 28.20        | 35.5    
resnet50           | 165.40     | 41.50      | 84.50        | 11.8    
==============================================================
```

---

## 🌐 4. Launching the Showcase Website

The website (`index.html`) is completely self-contained with:
- **Interactive Live Detector Playground**: Test detections on sample fruit images or upload custom fruit photos.
- **Dynamic HUD**: Real-time display of Latency (ms), FPS, and Bounding Box counts.
- **Interactive Pareto & Radar Charts**: Direct visualization of speed vs. accuracy trade-offs.
- **Edge Deployment Roadmap**: Guides for Raspberry Pi, Android, and WebAssembly.

### How to view:
1. Simply double-click `index.html` to open it in Chrome, Edge, or Firefox.
2. Or serve it locally with Python:
   ```bash
   python -m http.server 8000
   ```
   and navigate to `http://localhost:8000`.

---

## 📦 5. Exporting to ONNX for On-Device Deployment

Convert trained checkpoints to ONNX:
```bash
python export_onnx.py --model mobilenet_v3 --weights checkpoints/mobilenet_v3_best.pth
```
This `.onnx` model can be deployed directly with **ONNX Runtime Web**, **Android NNAPI**, or embedded on a **Raspberry Pi**.
