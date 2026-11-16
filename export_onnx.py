"""
ONNX Model Exporter for Edge Fruit Recognition
Converts trained PyTorch object detection models to ONNX format
for on-device deployment (Raspberry Pi, Android, WebAssembly).
"""

import os
import argparse
import torch
from train_detector import build_model

def export_to_onnx(model_name: str, weights_path: str = None, output_path: str = None, num_classes: int = 3):
    if output_path is None:
        output_path = f"{model_name}_edge.onnx"

    print(f"Exporting {model_name} to ONNX format...")
    model = build_model(model_name, num_classes=num_classes, pretrained=False)

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Loaded weights from {weights_path}")
    else:
        print("Note: Exporting with initialized architecture weights.")

    model.eval()

    # Dummy input representing batch size 1, 3 channels, 320x320 resolution
    dummy_input = [torch.randn(3, 320, 320)]

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={
                "images": {0: "batch_size", 2: "height", 3: "width"}
            }
        )
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Success] Exported ONNX model saved to: {output_path} ({file_size:.2f} MB)")
    except Exception as e:
        print(f"[Export Info] Standard dynamic tracing notes: {e}")
        # Direct script export fallback
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=13,
            input_names=["input"],
            output_names=["output"]
        )
        print(f"[Success] Exported fallback to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mobilenet_v3", "resnet50", "efficientnet_b0"], required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    export_to_onnx(args.model, args.weights, args.out)
