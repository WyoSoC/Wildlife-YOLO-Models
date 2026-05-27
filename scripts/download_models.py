"""
Download a Wildlife YOLO model from the UWyo HuggingFace collection.

Usage:
    python scripts/download_models.py --model golden_eagle --out models/
    python scripts/download_models.py --model all --out models/
    python scripts/download_models.py --list

After downloading, copy the .pt file into the Wildlife PTZ Camera Tracker
app's models/ directory — it will appear automatically in the UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HF_ORG = "UWyo"

MODEL_REGISTRY: dict[str, dict] = {
    "golden_eagle": {
        "repo_id": f"{HF_ORG}/wildlife-golden-eagle",
        "filename": "best.pt",
        "description": "Golden Eagle (Aquila chrysaetos) — single-class detection",
    },
    "pronghorn": {
        "repo_id": f"{HF_ORG}/wildlife-pronghorn",
        "filename": "best.pt",
        "description": "Pronghorn (Antilocapra americana) — single-class detection",
    },
    "bighorn_sheep": {
        "repo_id": f"{HF_ORG}/wildlife-bighorn-sheep",
        "filename": "best.pt",
        "description": "Bighorn Sheep (Ovis canadensis) — single-class detection",
    },
    "bison": {
        "repo_id": f"{HF_ORG}/wildlife-bison",
        "filename": "best.pt",
        "description": "American Bison (Bison bison) — single-class detection",
    },
    "north_american_wildlife": {
        "repo_id": f"{HF_ORG}/wildlife-north-american",
        "filename": "best.pt",
        "description": "Multi-species: Golden Eagle, Pronghorn, Bighorn Sheep, Bison, Mule Deer, Elk, Coyote",
    },
}


def list_models() -> None:
    print(f"{'Model':<30}  {'Description'}")
    print("-" * 80)
    for name, info in MODEL_REGISTRY.items():
        print(f"{name:<30}  {info['description']}")


def download(model_name: str, out_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print("[error] Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    info = MODEL_REGISTRY[model_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"{model_name}.pt"
    dest = out_dir / dest_name

    if dest.exists():
        print(f"[skip] {dest} already exists. Delete it to re-download.")
        return dest

    print(f"[download] {model_name} from {info['repo_id']} …")
    try:
        cached = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=str(out_dir),
        )
        # Rename to <species>.pt for clarity
        cached_path = Path(cached)
        if cached_path.name != dest_name:
            cached_path.rename(dest)
        print(f"[done] Saved to {dest}")
    except Exception as e:
        print(f"[error] Download failed: {e}", file=sys.stderr)
        print(
            f"       Model may not be published yet. Check: "
            f"https://huggingface.co/{info['repo_id']}",
            file=sys.stderr,
        )
        sys.exit(1)

    return dest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Wildlife YOLO models from HuggingFace"
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name (e.g. pronghorn) or 'all'. Use --list to see available models.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("models"),
        help="Output directory (default: models/)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        list_models()
        return

    if not args.model:
        print("[error] Specify --model <name> or --list", file=sys.stderr)
        sys.exit(1)

    if args.model == "all":
        for name in MODEL_REGISTRY:
            download(name, args.out)
    else:
        if args.model not in MODEL_REGISTRY:
            print(f"[error] Unknown model '{args.model}'", file=sys.stderr)
            print(f"        Available: {list(MODEL_REGISTRY.keys())}", file=sys.stderr)
            sys.exit(1)
        dest = download(args.model, args.out)
        print(f"\nCopy {dest} into the Wildlife PTZ Camera Tracker app's models/ folder.")


if __name__ == "__main__":
    main()
