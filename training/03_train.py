"""
Fine-tune YOLO11 (or YOLOv8) on a wildlife dataset.

Usage:
    python 03_train.py --config configs/pronghorn.yaml --base yolo11s.pt --device cuda
    python 03_train.py --config configs/north_american_wildlife.yaml --base yolo11m.pt --device mps
    python 03_train.py --config configs/pronghorn.yaml --base yolo11s.pt --device cuda --half

    # Resume interrupted training:
    python 03_train.py --config configs/pronghorn.yaml --resume runs/train/pronghorn/weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO11 on a wildlife dataset")
    p.add_argument("--config",  type=Path, required=True, help="YAML dataset config")
    p.add_argument("--base",    default="yolo11s.pt", help="Base weights (default: yolo11s.pt)")
    p.add_argument("--device",  default="cpu",         help="cpu | cuda | mps | 0,1,… (default: cpu)")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=-1,  help="-1 = auto-batch")
    p.add_argument("--lr0",     type=float, default=0.01)
    p.add_argument("--lrf",     type=float, default=0.01, help="Final LR = lr0 * lrf")
    p.add_argument("--half",    action="store_true",   help="FP16 training (CUDA only)")
    p.add_argument("--project", default="runs/train",  help="Output project dir")
    p.add_argument("--name",    default=None,           help="Run name (default: config stem)")
    p.add_argument("--resume",  default=None,           help="Resume from last.pt path")
    p.add_argument("--wandb",   action="store_true",    help="Enable Weights & Biases logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("[error] Install ultralytics: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    if args.wandb:
        try:
            import wandb  # type: ignore
            wandb.init(project="wildlife-yolo", name=args.name or args.config.stem)
        except ImportError:
            print("[warn] wandb not installed — disabling W&B logging.")

    run_name = args.name or args.config.stem

    if args.resume:
        model = YOLO(args.resume)
        print(f"[train] Resuming from {args.resume}")
    else:
        model = YOLO(args.base)
        print(f"[train] Base model: {args.base}")

    print(f"[train] Config:  {args.config}")
    print(f"[train] Device:  {args.device}")
    print(f"[train] Epochs:  {args.epochs}  |  imgsz: {args.imgsz}  |  batch: {args.batch}")

    train_kwargs = dict(
        data=str(args.config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        lrf=args.lrf,
        half=args.half,
        project=args.project,
        name=run_name,
        exist_ok=True,
        # Augmentation defaults appropriate for wildlife / camera-trap images
        mosaic=1.0,
        degrees=10.0,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    if args.resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)

    best_weights = Path(args.project) / run_name / "weights" / "best.pt"
    print(f"\n[done] Training complete.")
    print(f"       Best weights: {best_weights}")
    print(f"       mAP50:        {results.results_dict.get('metrics/mAP50(B)', 'n/a'):.4f}")
    print(f"       mAP50-95:     {results.results_dict.get('metrics/mAP50-95(B)', 'n/a'):.4f}")
    print(f"\nNext step:\n  python 04_evaluate.py --weights {best_weights} --data {args.config}")


if __name__ == "__main__":
    main()
