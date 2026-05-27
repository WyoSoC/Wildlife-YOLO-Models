"""
Export a trained YOLO model to ONNX, TensorRT, or CoreML.

Usage:
    # ONNX — works on any device (CPU / CUDA / Jetson):
    python 05_export.py --weights runs/train/pronghorn/weights/best.pt --format onnx

    # TensorRT engine — run ON the target Jetson; generates a .engine file:
    python 05_export.py --weights runs/train/pronghorn/weights/best.pt \
                        --format engine --device cuda --half

    # CoreML — macOS / iOS deployment:
    python 05_export.py --weights runs/train/pronghorn/weights/best.pt --format coreml

    # INT8 quantised ONNX (smallest, fastest on CPU edge devices):
    python 05_export.py --weights runs/train/pronghorn/weights/best.pt \
                        --format onnx --int8

Supported formats (subset that Ultralytics handles):
  onnx | engine (TensorRT) | coreml | openvino | tflite | saved_model | paddle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a trained wildlife YOLO model")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    p.add_argument(
        "--format",
        choices=["onnx", "engine", "coreml", "openvino", "tflite", "saved_model", "paddle"],
        default="onnx",
        help="Export format (default: onnx)",
    )
    p.add_argument("--imgsz",   type=int,  default=640,   help="Input image size (default: 640)")
    p.add_argument("--device",  default="cpu",             help="cpu | cuda | mps (default: cpu)")
    p.add_argument("--half",    action="store_true",       help="FP16 export (CUDA/TensorRT only)")
    p.add_argument("--int8",    action="store_true",       help="INT8 quantisation (ONNX / TensorRT)")
    p.add_argument("--batch",   type=int,  default=1,     help="Export batch size (default: 1)")
    p.add_argument("--simplify", action="store_true", default=True,
                   help="Simplify ONNX graph (default: True)")
    p.add_argument("--opset",   type=int,  default=17,   help="ONNX opset (default: 17)")
    p.add_argument("--dynamic", action="store_true",       help="Dynamic axes for ONNX")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("[error] Install ultralytics: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(args.weights))
    print(f"[export] Weights: {args.weights}")
    print(f"[export] Format:  {args.format}")
    print(f"[export] Device:  {args.device}  |  FP16: {args.half}  |  INT8: {args.int8}")

    export_kwargs: dict = dict(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        int8=args.int8,
        batch=args.batch,
    )

    if args.format == "onnx":
        export_kwargs.update(
            simplify=args.simplify,
            opset=args.opset,
            dynamic=args.dynamic,
        )

    exported_path = model.export(**export_kwargs)
    print(f"\n[done] Exported model: {exported_path}")

    if args.format == "engine":
        print("\nNote: TensorRT .engine files are device-specific.")
        print("      Generated on this machine; copy to target Jetson and verify.")
    elif args.format == "onnx":
        print("\nTip: run with onnxruntime for portability:")
        print("     pip install onnxruntime   # CPU")
        print("     pip install onnxruntime-gpu  # GPU")
    elif args.format == "coreml":
        print("\nCopy the .mlpackage to your Xcode project or use via coremltools.")


if __name__ == "__main__":
    main()
