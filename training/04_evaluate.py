"""
Evaluate a trained YOLO model on the test split.

Outputs:
  - mAP50 / mAP50-95
  - Per-class precision, recall, F1
  - Confusion matrix (saved as PNG)
  - Inference speed (ms/image on the evaluation device)

Usage:
    python 04_evaluate.py --weights runs/train/pronghorn/weights/best.pt \
                          --data configs/pronghorn.yaml

    # Evaluate at multiple confidence thresholds:
    python 04_evaluate.py --weights runs/train/pronghorn/weights/best.pt \
                          --data configs/pronghorn.yaml --conf-sweep

    # Benchmark inference speed only (no metric computation):
    python 04_evaluate.py --weights runs/train/pronghorn/weights/best.pt \
                          --data configs/pronghorn.yaml --speed-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained wildlife YOLO model")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    p.add_argument("--data",    type=Path, required=True, help="YAML dataset config")
    p.add_argument("--device",  default="cpu",            help="cpu | cuda | mps")
    p.add_argument("--imgsz",   type=int,  default=640)
    p.add_argument("--batch",   type=int,  default=16)
    p.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou",     type=float, default=0.50, help="NMS IoU threshold")
    p.add_argument("--split",   default="test",           help="Dataset split to evaluate (default: test)")
    p.add_argument("--save-json",   action="store_true",  help="Save COCO-format predictions JSON")
    p.add_argument("--save-plots",  action="store_true",  help="Save PR curve and confusion matrix")
    p.add_argument("--conf-sweep",  action="store_true",  help="Sweep conf thresholds and print F1 table")
    p.add_argument("--speed-only",  action="store_true",  help="Benchmark speed only (no mAP)")
    p.add_argument("--project", default="runs/eval",      help="Output directory")
    p.add_argument("--name",    default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("[error] Install ultralytics: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(args.weights))
    run_name = args.name or args.weights.parent.parent.name + "_eval"

    if args.speed_only:
        print("[speed] Benchmarking inference speed …")
        bench = model.benchmark(imgsz=args.imgsz, device=args.device, half=False)
        print(bench)
        return

    val_kwargs = dict(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        save_json=args.save_json,
        plots=args.save_plots,
        project=args.project,
        name=run_name,
        verbose=True,
    )

    metrics = model.val(**val_kwargs)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:        {args.weights}")
    print(f"  Dataset:      {args.data}  [{args.split}]")
    print(f"  Device:       {args.device}")
    print(f"  mAP50:        {metrics.box.map50:.4f}")
    print(f"  mAP50-95:     {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    speed = metrics.speed
    print(f"  Speed:        preprocess {speed['preprocess']:.1f} ms | "
          f"inference {speed['inference']:.1f} ms | "
          f"postprocess {speed['postprocess']:.1f} ms")

    if args.conf_sweep:
        import numpy as np
        print("\n  Confidence sweep:")
        print(f"  {'conf':>6}  {'P':>7}  {'R':>7}  {'F1':>7}")
        for conf in np.arange(0.10, 0.91, 0.10):
            m = model.val(
                data=str(args.data),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                conf=round(float(conf), 2),
                iou=args.iou,
                split=args.split,
                verbose=False,
            )
            prec = m.box.mp
            rec  = m.box.mr
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            print(f"  {conf:.2f}    {prec:.4f}   {rec:.4f}   {f1:.4f}")

    print(f"\n[done] Artefacts saved to: {args.project}/{run_name}/")


if __name__ == "__main__":
    main()
