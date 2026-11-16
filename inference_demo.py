"""
Fruit Detection Inference Script
Loads the trained pilot models (MobileNetV3, EfficientNet-B0, ResNet-50)
Trained classes: Dragon Fruit, Watermelon
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["__background__", "Dragon Fruit", "Watermelon"]
CLASS_COLORS = {
    "Dragon Fruit": "#d946ef",
    "Watermelon": "#16a34a"
}

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

def predict(image_path: str, model_name: str = "mobilenetv3", weights_path: str = None, conf_threshold: float = 0.40, output_path: str = "output_detection.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Fruit Detector] Loading {model_name} on {device}...")

    if weights_path is None:
        weights_path = f"trained_models/{model_name}_best.pth"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    model = build_model(model_name, num_classes=len(CLASSES))
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    tensor_img = (pil_to_tensor(image).float() / 255.0).to(device)

    # Time the actual inference
    t0 = time.perf_counter()
    with torch.no_grad():
        predictions = model([tensor_img])[0]
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # Filter detections and count fruits
    fruit_counts = {"Dragon Fruit": 0, "Watermelon": 0}
    draw = ImageDraw.Draw(image)

    detected_count = 0
    print(f"\n--- Detection Results ({latency_ms:.2f} ms latency) ---")
    for box, label_idx, score in zip(boxes, labels, scores):
        if score >= conf_threshold and label_idx < len(CLASSES):
            class_name = CLASSES[label_idx]
            fruit_counts[class_name] = fruit_counts.get(class_name, 0) + 1
            detected_count += 1

            color = CLASS_COLORS.get(class_name, "#10b981")
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label_text = f"{class_name}: {score * 100:.1f}%"
            draw.text((x1 + 4, max(4, y1 - 16)), label_text, fill=color)
            print(f"  • {class_name} ({score * 100:.1f}%) at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    print("\n--- Fruit Count Summary ---")
    for fruit, count in fruit_counts.items():
        print(f"  • {fruit}: {count}")
    print(f"  • Total Fruits Detected: {detected_count}")

    image.save(output_path)
    print(f"\nAnnotated result saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input fruit image")
    parser.add_argument("--model", type=str, choices=["mobilenetv3", "efficientnet_b0", "resnet50"], default="mobilenetv3")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth checkpoint file")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output_detection.jpg", help="Output annotated image path")
    args = parser.parse_args()

    predict(
        image_path=args.image,
        model_name=args.model,
        weights_path=args.weights,
        conf_threshold=args.conf,
        output_path=args.output
    )
