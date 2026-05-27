"""
Prepare raw camera-trap images for YOLO training.

Pipeline:
  1. (Optional) Run MegaDetector to generate animal bounding boxes.
  2. Filter detections: keep "animal" class above confidence threshold.
  3. Convert bboxes to YOLO format (cx, cy, w, h — normalised).
  4. Split into train / val / test sets.
  5. Write images + labels into the directory layout expected by the YAML configs.

Usage:
    # With MegaDetector bbox generation (recommended for iNaturalist data):
    python 02_prepare_dataset.py --species golden_eagle --dataset ../datasets/ --megadetector

    # Using pre-existing COCO-format annotations (NACTI / WCS):
    python 02_prepare_dataset.py --species pronghorn --dataset ../datasets/

    # Multi-species combined dataset:
    python 02_prepare_dataset.py --species all --dataset ../datasets/ --megadetector

    # Oversample minority classes to address class imbalance:
    python 02_prepare_dataset.py --species all --dataset ../datasets/ --balance
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIES_CLASS_MAP: dict[str, int] = {
    "golden_eagle":  0,
    "pronghorn":     1,
    "bighorn_sheep": 2,
    "bison":         3,
    "mule_deer":     4,
    "elk":           5,
    "coyote":        6,
}

MEGADETECTOR_ANIMAL_CLASS = 1  # MegaDetector: 1=animal, 2=human, 3=vehicle
MEGADETECTOR_CONF_THRESHOLD = 0.15  # low threshold; filter further if needed

SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / val / test


# ---------------------------------------------------------------------------
# MegaDetector
# ---------------------------------------------------------------------------

def run_megadetector(
    image_dir: Path,
    output_json: Path,
    batch_size: int = 8,
    device: str = "cpu",
) -> None:
    """Run MegaDetector v5 on all images in image_dir; write results to output_json."""
    try:
        from megadetector.detection.run_detector_batch import (  # type: ignore
            load_and_run_detector_batch,
            write_results_to_file,
        )
    except ImportError:
        raise RuntimeError(
            "MegaDetector not installed. Run:\n"
            "  pip install git+https://github.com/agentmorris/MegaDetector.git"
        )

    images = sorted(image_dir.rglob("*.jpg")) + sorted(image_dir.rglob("*.png"))
    image_paths = [str(p) for p in images]

    results = load_and_run_detector_batch(
        model_file="MDV5A",  # auto-downloads MDv5a weights
        image_file_names=image_paths,
        checkpoint_path=str(output_json.with_suffix(".ckpt")),
        confidence_threshold=MEGADETECTOR_CONF_THRESHOLD,
        batch_size=batch_size,
        device=device,
    )
    write_results_to_file(results, str(output_json))
    print(f"[megadetector] Results written to {output_json}")


# ---------------------------------------------------------------------------
# COCO annotation parsing
# ---------------------------------------------------------------------------

def load_coco_annotations(ann_file: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    """Return (id→image_info, image_id→[annotation, …]) from a COCO JSON."""
    with open(ann_file) as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco.get("images", [])}
    anns: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns.setdefault(ann["image_id"], []).append(ann)
    return images, anns


# ---------------------------------------------------------------------------
# YOLO label helpers
# ---------------------------------------------------------------------------

def bbox_to_yolo(bbox_xywh: list[float], img_w: int, img_h: int) -> tuple[float, ...]:
    """Convert COCO [x, y, w, h] (absolute pixels) to YOLO normalised cx cy w h."""
    x, y, w, h = bbox_xywh
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)


def megadetector_to_yolo(
    det: dict, img_w: int, img_h: int
) -> tuple[float, ...]:
    """
    Convert a MegaDetector detection to YOLO normalised cx cy w h.
    MegaDetector uses [x1, y1, w, h] relative [0–1].
    """
    x1, y1, w, h = det["bbox"]
    cx = x1 + w / 2
    cy = y1 + h / 2
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def write_label(label_path: Path, class_id: int, *boxes: tuple[float, ...]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{class_id} {' '.join(str(v) for v in box)}" for box in boxes]
    label_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Dataset preparation per species
# ---------------------------------------------------------------------------

def prepare_single_species(
    species: str,
    dataset_root: Path,
    out_root: Path,
    use_megadetector: bool,
    device: str,
    multi_class: bool,
) -> list[tuple[Path, Path]]:
    """
    Prepare images + labels for one species.
    Returns list of (image_path, label_path) pairs (absolute, in out_root).
    """
    class_id = SPECIES_CLASS_MAP[species] if multi_class else 0
    raw_dir = dataset_root / species / "raw"
    label_cache = dataset_root / species / "labels_cache"
    label_cache.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []

    if use_megadetector:
        md_json = dataset_root / species / "megadetector_results.json"
        if not md_json.exists():
            run_megadetector(raw_dir, md_json, device=device)

        with open(md_json) as f:
            md_results = json.load(f)

        for item in tqdm(md_results.get("images", []), desc=f"[md] {species}"):
            dets = [
                d for d in item.get("detections", [])
                if d.get("category") == str(MEGADETECTOR_ANIMAL_CLASS)
                and d.get("conf", 0) >= MEGADETECTOR_CONF_THRESHOLD
            ]
            if not dets:
                continue

            img_path = Path(item["file"])
            if not img_path.exists():
                img_path = raw_dir / img_path.name
            if not img_path.exists():
                continue

            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                continue

            label_path = label_cache / (img_path.stem + ".txt")
            boxes = [megadetector_to_yolo(d, img_w, img_h) for d in dets]
            write_label(label_path, class_id, *boxes)
            pairs.append((img_path, label_path))

    else:
        # Look for COCO annotation files
        for ann_file in (dataset_root / species).glob("*.json"):
            images_meta, ann_map = load_coco_annotations(ann_file)
            for img_id, img_info in tqdm(
                images_meta.items(), desc=f"[coco] {species}/{ann_file.name}"
            ):
                img_anns = ann_map.get(img_id, [])
                if not img_anns:
                    continue

                img_path = raw_dir / img_info["file_name"]
                if not img_path.exists():
                    continue

                img_w = img_info["width"]
                img_h = img_info["height"]

                label_path = label_cache / (img_path.stem + ".txt")
                boxes = [
                    bbox_to_yolo(a["bbox"], img_w, img_h)
                    for a in img_anns
                    if a.get("bbox")
                ]
                if boxes:
                    write_label(label_path, class_id, *boxes)
                    pairs.append((img_path, label_path))

        # Fallback: raw images with no annotation (generate synthetic whole-image bbox)
        if not pairs:
            print(f"[warn] No COCO annotations for '{species}'. Generating whole-image labels.")
            for img_path in tqdm(sorted(raw_dir.glob("*.jpg")), desc=f"[raw] {species}"):
                label_path = label_cache / (img_path.stem + ".txt")
                write_label(label_path, class_id, (0.5, 0.5, 1.0, 1.0))
                pairs.append((img_path, label_path))

    return pairs


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_and_copy(
    pairs: list[tuple[Path, Path]],
    out_root: Path,
    seed: int = 42,
    oversample_factor: int = 1,
) -> None:
    """Shuffle, split, copy image+label pairs into train/val/test directories."""
    random.seed(seed)
    shuffled = list(pairs) * oversample_factor
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        img_dir = out_root / "images" / split_name
        lbl_dir = out_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_src, lbl_src in tqdm(split_pairs, desc=f"  copy → {split_name}"):
            shutil.copy2(img_src, img_dir / img_src.name)
            shutil.copy2(lbl_src, lbl_dir / lbl_src.stem + ".txt")

    for split_name, split_pairs in splits.items():
        print(f"  {split_name}: {len(split_pairs)} samples")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare wildlife dataset for YOLO training")
    p.add_argument(
        "--species",
        nargs="+",
        default=["pronghorn"],
        help="Species to prepare. Use 'all' for every configured species.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("../datasets"),
        help="Root directory containing per-species raw images and annotations",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (default: --dataset/<species> or --dataset/north_american_wildlife)",
    )
    p.add_argument(
        "--megadetector",
        action="store_true",
        help="Run MegaDetector to generate bboxes (required for iNaturalist data)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for MegaDetector inference: cpu | cuda | mps (default: cpu)",
    )
    p.add_argument(
        "--balance",
        action="store_true",
        help="Oversample minority classes to address class imbalance",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_species = list(SPECIES_CLASS_MAP.keys())
    species_list: list[str] = (
        all_species if "all" in args.species else args.species
    )
    unknown = [s for s in species_list if s not in SPECIES_CLASS_MAP]
    if unknown:
        import sys
        print(f"[error] Unknown species: {unknown}", file=sys.stderr)
        sys.exit(1)

    multi_class = len(species_list) > 1
    dataset_root = args.dataset.resolve()

    if multi_class:
        out_name = "north_american_wildlife"
    else:
        out_name = species_list[0]

    out_root = (args.out or dataset_root / out_name).resolve()
    print(f"[info] Output dataset: {out_root}")
    print(f"[info] Species: {species_list}")
    print(f"[info] MegaDetector: {args.megadetector}")
    print(f"[info] Multi-class: {multi_class}")

    # Collect pairs per species
    all_pairs: list[tuple[Path, Path]] = []
    species_counts: dict[str, int] = {}
    for sp in species_list:
        pairs = prepare_single_species(
            sp, dataset_root, out_root,
            use_megadetector=args.megadetector,
            device=args.device,
            multi_class=multi_class,
        )
        species_counts[sp] = len(pairs)
        all_pairs.extend(pairs)
        print(f"  {sp}: {len(pairs)} annotated images")

    if args.balance and multi_class:
        max_count = max(species_counts.values())
        balanced: list[tuple[Path, Path]] = []
        sp_pairs: dict[str, list] = {sp: [] for sp in species_list}
        # Re-partition by species
        for img_path, lbl_path in all_pairs:
            for sp in species_list:
                if (dataset_root / sp).as_posix() in img_path.as_posix():
                    sp_pairs[sp].append((img_path, lbl_path))
                    break
        for sp, pairs in sp_pairs.items():
            factor = max(1, round(max_count / len(pairs))) if pairs else 1
            balanced.extend(pairs * factor)
            print(f"  [balance] {sp}: {len(pairs)} × {factor} = {len(pairs) * factor}")
        all_pairs = balanced

    print(f"\n[info] Total pairs: {len(all_pairs)}")
    split_and_copy(all_pairs, out_root, seed=args.seed)
    print(f"\n[done] Dataset ready at {out_root}")


if __name__ == "__main__":
    main()
