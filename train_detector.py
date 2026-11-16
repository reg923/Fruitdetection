"""
Training Script for Fruit Object Detection
Compares 3 Architectures:
  1. MobileNetV3-Large (SSDLite) - Lightweight Edge Champion
  2. ResNet-50 v2 (Faster R-CNN FPN) - High Precision Baseline
  3. EfficientNet-B0 (Faster R-CNN) - Balanced Speed & Accuracy
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from dataset_loader import FruitBoundingBoxDataset, collate_fn

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Builds the detection model for custom fruit classes.
    num_classes: number of fruit classes + 1 (background)
    """
    if model_name == "mobilenet_v3":
        if pretrained:
            model = ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
                num_classes=num_classes
            )
        else:
            model = ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes
            )
        return model

    elif model_name == "resnet50":
        if pretrained:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        else:
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes
            )
        return model

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights).features
        base.out_channels = 1280
        
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        model = FasterRCNN(
            backbone=base,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_detector(
    data_dir: str,
    model_name: str,
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 0.0005,
    num_classes: int = 3, # 2 fruits + background
    output_dir: str = "./checkpoints"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f" Starting Detection Training: {model_name.upper()}")
    print(f" Target Device: {device} | Total Classes: {num_classes} (incl. background)")
    print(f"==========================================\n")

    train_dataset = FruitBoundingBoxDataset(root_dir=data_dir, split="train")

    if len(train_dataset) == 0:
        print(f"[Notice] No training images detected in: {data_dir}/images/train")
        print("Dataset template is ready! Once you move your annotated images into")
        print(f"  {data_dir}/images/train and {data_dir}/labels/train")
        print("re-run this script to train your detector.\n")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = build_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s - LR: {lr_scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >>> Best model saved to: {ckpt_path}")

    print(f"\n[Completed] {model_name} training finished. Best Training Loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fruit Recognition Object Detectors")
    parser.add_argument("--data_dir", type=str, default="./fruit_dataset", help="Path to fruit dataset root")
    parser.add_argument("--model", type=str, choices=["mobilenet_v3", "resnet50", "efficientnet_b0"], required=True)
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (reduce if GPU OOM)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of fruit classes + 1 background")
    args = parser.parse_args()

    train_detector(
        data_dir=args.data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=args.num_classes
    )
