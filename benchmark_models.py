"""
Benchmark Tool for Edge Fruit Recognition
Measures:
  - Model weights file size (MB)
  - Trainable parameter count (M)
  - Inference Latency (ms) on CPU & Edge simulation
  - Frames Per Second (FPS)
"""

import os
import time
import argparse
import torch
from train_detector import build_model

def run_benchmark(model_name: str, weights_path: str = None, num_classes: int = 3, runs: int = 100):
    device = torch.device("cpu") # Benchmarking on CPU simulates typical low-cost edge device
    print(f"\n=======================================================")
    print(f" BENCHMARKING: {model_name.upper()} on {device}")
    print(f"=======================================================")

    # Initialize model
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f" Loaded weights from: {weights_path}")
    else:
        # Measure in-memory state dict size if weights file not yet trained
        temp_path = f"temp_{model_name}.pth"
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        if os.path.exists(temp_path):
            os.remove(temp_path)

    model.eval()
    model.to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Prepare dummy input (1, 3, 320, 320)
    dummy_input = [torch.randn(3, 320, 320, device=device)]

    # Warmup runs
    for _ in range(15):
        with torch.no_grad():
            _ = model(dummy_input)

    # Timed inference iterations
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    fps = 1000.0 / avg_latency

    print(f"\n--- Benchmark Results for {model_name} ---")
    print(f"  • Model Size on Disk:      {size_mb:.2f} MB")
    print(f"  • Parameter Count:         {total_params / 1e6:.2f} Million")
    print(f"  • Mean Inference Latency:  {avg_latency:.2f} ms")
    print(f"  • Min / Max Latency:       {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f"  • Throughput:              {fps:.1f} FPS")

    # Edge Viability Scoring
    if avg_latency < 30 and size_mb < 30:
        badge = "EXCELLENT FOR EDGE (Microcontroller / Mobile / Raspberry Pi)"
    elif avg_latency < 60 and size_mb < 60:
        badge = "BALANCED EDGE / MOBILE (Suitable for mid-range smartphones / Jetson)"
    else:
        badge = "HEAVY BASELINE (Requires Workstation / Cloud Server or GPU acceleration)"
    print(f"  • Edge Viability:          {badge}\n")

    return {
        "model": model_name,
        "size_mb": size_mb,
        "params_m": total_params / 1e6,
        "latency_ms": avg_latency,
        "fps": fps
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mobilenet_v3", "resnet50", "efficientnet_b0", "all"], default="all")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    models_to_test = ["mobilenet_v3", "efficientnet_b0", "resnet50"] if args.model == "all" else [args.model]
    
    summary = []
    for m in models_to_test:
        summary.append(run_benchmark(m, args.weights))

    print("\n================== EXECUTIVE SUMMARY MATRIX ==================")
    print(f"{'Model':<18} | {'Size (MB)':<10} | {'Params (M)':<10} | {'Latency (ms)':<12} | {'FPS':<8}")
    print("-" * 68)
    for row in summary:
        print(f"{row['model']:<18} | {row['size_mb']:<10.2f} | {row['params_m']:<10.2f} | {row['latency_ms']:<12.2f} | {row['fps']:<8.1f}")
    print("==============================================================\n")
