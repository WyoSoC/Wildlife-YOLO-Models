"""
Upload trained model weights and model card to the HuggingFace Hub.

Usage:
    huggingface-cli login   # one-time auth

    python 06_upload_to_hf.py \
        --weights runs/train/pronghorn/weights/best.pt \
        --model-id WyoSoC/wildlife-pronghorn \
        --species pronghorn

    # Also upload ONNX:
    python 06_upload_to_hf.py \
        --weights runs/train/pronghorn/weights/best.pt \
        --onnx    runs/train/pronghorn/weights/best.onnx \
        --model-id WyoSoC/wildlife-pronghorn \
        --species pronghorn \
        --private   # keep private until review
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

SPECIES_META: dict[str, dict] = {
    "golden_eagle":  {"latin": "Aquila chrysaetos",   "base": "YOLO11s"},
    "pronghorn":     {"latin": "Antilocapra americana", "base": "YOLO11s"},
    "bighorn_sheep": {"latin": "Ovis canadensis",      "base": "YOLO11s"},
    "bison":         {"latin": "Bison bison",           "base": "YOLO11s"},
    "mule_deer":     {"latin": "Odocoileus hemionus",   "base": "YOLO11s"},
    "elk":           {"latin": "Cervus canadensis",     "base": "YOLO11s"},
    "coyote":        {"latin": "Canis latrans",         "base": "YOLO11s"},
    "north_american_wildlife": {"latin": "multi-species (7 classes)", "base": "YOLO11m"},
}


def generate_model_card(
    species: str,
    model_id: str,
    map50: float | None,
    map50_95: float | None,
) -> str:
    meta = SPECIES_META.get(species, {"latin": species, "base": "YOLO11s"})
    map50_str    = f"{map50:.4f}" if map50 is not None else "TBD"
    map50_95_str = f"{map50_95:.4f}" if map50_95 is not None else "TBD"

    return f"""---
license: cc-by-4.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - wildlife
  - yolo
  - yolo11
  - {species.replace("_", "-")}
  - camera-trap
  - edge-ai
---

# Wildlife YOLO — {species.replace("_", " ").title()} ({meta["latin"]})

YOLO11 object detection model fine-tuned for **{species.replace("_", " ")}** detection
in camera-trap and PTZ camera footage.

Part of the [WyoSoC Wildlife YOLO Models](https://github.com/WyoSoC/Wildlife-YOLO-Models)
collection. Drop the `.pt` file into the
[Wildlife PTZ Camera Tracker](https://github.com/WyoSoC/Wildlife_PTZ_Camera_Tracker)
app's `models/` folder to activate it.

## Model Details

| Property | Value |
|---|---|
| Base architecture | {meta["base"]} |
| Input size | 640 × 640 |
| Classes | 1 ({species.replace("_", " ")}) |
| mAP50 (test) | {map50_str} |
| mAP50-95 (test) | {map50_95_str} |
| Training data | NACTI / iNaturalist (GBIF) |

## Usage

```python
from ultralytics import YOLO

model = YOLO("WyoSoC/wildlife-{species.replace("_", "-")}")

results = model.predict("your_image.jpg", conf=0.25)
results[0].show()
```

Or download the weights directly:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('{model_id}', 'best.pt', local_dir='models/')
"
```

## Training

See the [training pipeline](https://github.com/WyoSoC/Wildlife-YOLO-Models/tree/main/training)
for data preparation, fine-tuning, and export instructions.

## License

CC-BY-4.0 — same as the underlying training datasets.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload YOLO model weights to HuggingFace Hub")
    p.add_argument("--weights",  type=Path, required=True, help="Path to best.pt")
    p.add_argument("--model-id", required=True,             help="HF repo, e.g. WyoSoC/wildlife-pronghorn")
    p.add_argument("--species",  required=True,             help="Species key, e.g. pronghorn")
    p.add_argument("--onnx",     type=Path, default=None,   help="Also upload ONNX weights")
    p.add_argument("--map50",    type=float, default=None,  help="mAP50 to embed in model card")
    p.add_argument("--map50-95", type=float, default=None,  dest="map50_95",
                   help="mAP50-95 to embed in model card")
    p.add_argument("--private",  action="store_true",       help="Create a private HF repo")
    p.add_argument("--commit-message", default="Upload trained YOLO model weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore
    except ImportError:
        print("[error] Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(args.model_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"[hf] Repo: https://huggingface.co/{args.model_id}")

    # Generate and upload model card
    card = generate_model_card(args.species, args.model_id, args.map50, args.map50_95)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(card)
        card_path = Path(f.name)

    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.model_id,
        commit_message=args.commit_message,
    )
    card_path.unlink(missing_ok=True)
    print("[hf] Model card uploaded.")

    # Upload PyTorch weights
    api.upload_file(
        path_or_fileobj=str(args.weights),
        path_in_repo="best.pt",
        repo_id=args.model_id,
        commit_message=args.commit_message,
    )
    print(f"[hf] Uploaded {args.weights.name} → best.pt")

    # Optionally upload ONNX
    if args.onnx and args.onnx.exists():
        api.upload_file(
            path_or_fileobj=str(args.onnx),
            path_in_repo="best.onnx",
            repo_id=args.model_id,
            commit_message=args.commit_message,
        )
        print(f"[hf] Uploaded {args.onnx.name} → best.onnx")

    print(f"\n[done] Model published at https://huggingface.co/{args.model_id}")


if __name__ == "__main__":
    main()
