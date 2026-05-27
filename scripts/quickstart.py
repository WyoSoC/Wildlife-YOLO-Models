"""
End-to-end quickstart: download → bbox → prepare → train → export.

Downloads iNaturalist images (no auth required), runs MegaDetector for
bounding-box generation, prepares a YOLO dataset, fine-tunes YOLO11n on
the local device, and exports the weights to ONNX.

Usage:
    # Single species, 200 images, quick 30-epoch run:
    python scripts/quickstart.py --species golden_eagle --max-images 200 --epochs 30

    # All 7 species, 150 images each, multi-class model:
    python scripts/quickstart.py --species all --max-images 150 --epochs 50

    # Resume after interruption:
    python scripts/quickstart.py --species pronghorn --resume

Options:
    --species       Species key(s) or 'all'  (default: golden_eagle pronghorn bison)
    --max-images    Images per species from iNaturalist  (default: 300)
    --epochs        Training epochs  (default: 50)
    --base          Base YOLO model weights  (default: yolo11n.pt — fastest)
    --device        cpu | cuda | mps  (default: auto-detect)
    --out           Root output dir  (default: ../datasets)
    --resume        Skip download/prep if data already exists
    --export-onnx   Also export ONNX after training  (default: True)
    --export-coreml Also export CoreML  (default: False)
    --no-megadetector  Skip MegaDetector; use whole-image bbox fallback
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── species registry ─────────────────────────────────────────────────────────

SPECIES_CONFIG: dict[str, dict] = {
    # ── original 7 ────────────────────────────────────────────────────────────
    "golden_eagle":    {"latin": "Aquila chrysaetos",          "taxon_id": 2480506, "class_id":  0},
    "pronghorn":       {"latin": "Antilocapra americana",      "taxon_id": 2440902, "class_id":  1},
    "bighorn_sheep":   {"latin": "Ovis canadensis",            "taxon_id": 2441119, "class_id":  2},
    "bison":           {"latin": "Bison bison",                "taxon_id": 2441176, "class_id":  3},
    "mule_deer":       {"latin": "Odocoileus hemionus",        "taxon_id": 2440974, "class_id":  4},
    "elk":             {"latin": "Cervus canadensis",          "taxon_id": 8600904, "class_id":  5},
    "coyote":          {"latin": "Canis latrans",              "taxon_id": 5219153, "class_id":  6},
    # ── new 18 ────────────────────────────────────────────────────────────────
    "grizzly_bear":    {"latin": "Ursus arctos horribilis",    "taxon_id": 6163845, "class_id":  7},
    "gray_wolf":       {"latin": "Canis lupus",                "taxon_id": 5219173, "class_id":  8},
    "moose":           {"latin": "Alces alces",                "taxon_id": 2440940, "class_id":  9},
    "pika":            {"latin": "Ochotona princeps",          "taxon_id": 2436982, "class_id": 10},
    "swift_fox":       {"latin": "Vulpes velox",               "taxon_id": 5219290, "class_id": 11},
    "mountain_lion":   {"latin": "Puma concolor",              "taxon_id": 2435099, "class_id": 12},
    "river_otter":     {"latin": "Lontra canadensis",          "taxon_id": 2433727, "class_id": 13},
    "black_bear":      {"latin": "Ursus americanus",           "taxon_id": 2433407, "class_id": 14},
    "bald_eagle":      {"latin": "Haliaeetus leucocephalus",   "taxon_id": 2480446, "class_id": 15},
    "red_tailed_hawk": {"latin": "Buteo jamaicensis",          "taxon_id": 2480542, "class_id": 16},
    "osprey":          {"latin": "Pandion haliaetus",          "taxon_id": 2480726, "class_id": 17},
    "sage_grouse":     {"latin": "Centrocercus urophasianus",  "taxon_id": 5959240, "class_id": 18},
    "trumpeter_swan":  {"latin": "Cygnus buccinator",         "taxon_id": 2498345, "class_id": 19},
    "beaver":          {"latin": "Castor canadensis",          "taxon_id": 2439838, "class_id": 20},
    "raven":           {"latin": "Corvus corax",               "taxon_id": 2482492, "class_id": 21},
    "prairie_dog":     {"latin": "Cynomys ludovicianus",       "taxon_id": 2437232, "class_id": 22},
    "badger":          {"latin": "Taxidea taxus",              "taxon_id": 2434102, "class_id": 23},
    "bobcat":          {"latin": "Lynx rufus",                 "taxon_id": 2435246, "class_id": 24},
}

SPLIT_RATIOS = (0.80, 0.10, 0.10)
MEGADETECTOR_ANIMAL_CLASS = 1
MEGADETECTOR_CONF = 0.15


# ── helpers ───────────────────────────────────────────────────────────────────

def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── step 1: download from iNaturalist GBIF ────────────────────────────────────

def download_inaturalist(species: str, raw_dir: Path, max_images: int) -> int:
    import requests

    taxon_id = SPECIES_CONFIG[species]["taxon_id"]
    mkdir(raw_dir)
    log("inaturalist", f"Fetching {species} (taxon {taxon_id}), target {max_images} images …")

    GBIF_API = "https://api.gbif.org/v1/occurrence/search"
    count, offset = 0, 0

    while count < max_images:
        try:
            resp = requests.get(GBIF_API, params={
                "taxonKey": taxon_id,
                "datasetKey": "50c9509d-22c7-4a22-a47d-8c48425ef4a7",
                "qualityGrade": "research",
                "mediaType": "StillImage",
                "limit": 100,
                "offset": offset,
            }, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            log("inaturalist", f"GBIF request failed: {e}. Retrying in 5 s …")
            time.sleep(5)
            continue

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for occ in results:
            if count >= max_images:
                break
            for media in occ.get("media", []):
                img_url = media.get("identifier")
                if not img_url:
                    continue
                dest = raw_dir / f"{occ['key']}.jpg"
                if dest.exists():
                    count += 1
                    break
                try:
                    r = requests.get(img_url, timeout=15)
                    if r.status_code == 200:
                        dest.write_bytes(r.content)
                        count += 1
                        if count % 25 == 0:
                            log("inaturalist", f"  {count}/{max_images} images downloaded")
                except Exception:
                    pass
                break

        offset += 100
        if data.get("endOfRecords"):
            break

    log("inaturalist", f"{species}: {count} images → {raw_dir}")
    return count


# ── step 2: MegaDetector bbox generation ─────────────────────────────────────

def run_megadetector(raw_dir: Path, out_json: Path, device: str) -> None:
    from megadetector.detection.run_detector_batch import (
        load_and_run_detector_batch,
        write_results_to_file,
    )

    images = sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.png"))
    if not images:
        log("megadetector", f"No images in {raw_dir} — skipping.")
        return

    log("megadetector", f"Running on {len(images)} images in {raw_dir} …")

    # MegaDetector auto-selects GPU/MPS when available; force_cpu=False (default)
    results = load_and_run_detector_batch(
        model_file="MDV5A",
        image_file_names=[str(p) for p in images],
        checkpoint_path=str(out_json.with_suffix(".ckpt")),
        confidence_threshold=MEGADETECTOR_CONF,
        batch_size=2,
    )
    write_results_to_file(results, str(out_json))
    log("megadetector", f"Results → {out_json}")


# ── step 3: build YOLO label files ────────────────────────────────────────────

def md_bbox_to_yolo(det: dict) -> tuple[float, float, float, float]:
    """MegaDetector [x1,y1,w,h] relative → YOLO cx,cy,w,h normalised."""
    x1, y1, w, h = det["bbox"]
    return round(x1 + w / 2, 6), round(y1 + h / 2, 6), round(w, 6), round(h, 6)


def build_labels_from_megadetector(
    raw_dir: Path,
    md_json: Path,
    label_cache: Path,
    class_id: int,
) -> list[tuple[Path, Path]]:
    mkdir(label_cache)
    with open(md_json) as f:
        md = json.load(f)

    pairs: list[tuple[Path, Path]] = []
    for item in md.get("images", []):
        dets = [
            d for d in item.get("detections", [])
            if d.get("category") == str(MEGADETECTOR_ANIMAL_CLASS)
            and d.get("conf", 0) >= MEGADETECTOR_CONF
        ]
        if not dets:
            continue

        img_path = Path(item["file"])
        if not img_path.exists():
            img_path = raw_dir / img_path.name
        if not img_path.exists():
            continue

        label_path = label_cache / (img_path.stem + ".txt")
        lines = [f"{class_id} {' '.join(str(v) for v in md_bbox_to_yolo(d))}" for d in dets]
        label_path.write_text("\n".join(lines) + "\n")
        pairs.append((img_path, label_path))

    return pairs


def build_labels_whole_image(
    raw_dir: Path,
    label_cache: Path,
    class_id: int,
) -> list[tuple[Path, Path]]:
    """Fallback: whole-image bounding box (0.5 0.5 1.0 1.0)."""
    mkdir(label_cache)
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(raw_dir.glob("*.jpg")):
        label_path = label_cache / (img_path.stem + ".txt")
        label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")
        pairs.append((img_path, label_path))
    return pairs


# ── step 4: split and copy into YOLO directory layout ─────────────────────────

def split_and_copy(
    pairs: list[tuple[Path, Path]],
    out_root: Path,
    seed: int = 42,
) -> None:
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val   = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        img_dir = mkdir(out_root / "images" / split_name)
        lbl_dir = mkdir(out_root / "labels" / split_name)
        for img_src, lbl_src in split_pairs:
            shutil.copy2(img_src, img_dir / img_src.name)
            shutil.copy2(lbl_src, lbl_dir / (lbl_src.stem + ".txt"))
        log("split", f"{split_name}: {len(split_pairs)} samples")


# ── step 5: write YAML config ─────────────────────────────────────────────────

def write_yaml_config(
    out_root: Path,
    species_list: list[str],
    config_path: Path,
) -> None:
    multi = len(species_list) > 1
    if multi:
        names = {SPECIES_CONFIG[sp]["class_id"]: sp for sp in species_list}
        nc = len(names)
    else:
        names = {0: species_list[0]}
        nc = 1

    lines = [
        f"path: {out_root}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        f"nc: {nc}",
        "names:",
    ]
    for cid in sorted(names):
        lines.append(f"  {cid}: {names[cid]}")

    config_path.write_text("\n".join(lines) + "\n")
    log("yaml", f"Config written to {config_path}")


# ── step 6: train ─────────────────────────────────────────────────────────────

def train(
    config_path: Path,
    base: str,
    epochs: int,
    device: str,
    run_name: str,
    project_dir: Path,
) -> Path:
    from ultralytics import YOLO

    log("train", f"Base: {base}  |  epochs: {epochs}  |  device: {device}")
    model = YOLO(base)
    model.train(
        data=str(config_path),
        epochs=epochs,
        imgsz=640,
        batch=-1,
        device=device,
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        degrees=10.0,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )
    best_pt = project_dir / run_name / "weights" / "best.pt"
    log("train", f"Best weights: {best_pt}")
    return best_pt


# ── step 7: export ────────────────────────────────────────────────────────────

def export_model(weights: Path, fmt: str, device: str) -> None:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    kwargs: dict = {"format": fmt, "imgsz": 640, "device": device}
    if fmt == "onnx":
        kwargs.update(simplify=True, opset=17)
    out = model.export(**kwargs)
    log("export", f"{fmt} → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wildlife YOLO quickstart pipeline")
    p.add_argument(
        "--species", nargs="+",
        default=["golden_eagle", "pronghorn", "bison"],
        help="Species or 'all'",
    )
    p.add_argument("--max-images", type=int, default=300,
                   help="Images per species from iNaturalist (default 300)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs (default 50)")
    p.add_argument("--base", default="yolo11n.pt",
                   help="Base weights (default yolo11n.pt — nano, fastest)")
    p.add_argument("--device", default=None,
                   help="cpu|cuda|mps (auto-detect if omitted)")
    p.add_argument("--out", type=Path, default=None,
                   help="Dataset root (default: <repo>/../datasets)")
    p.add_argument("--resume", action="store_true",
                   help="Skip download/prep if data already present")
    p.add_argument("--export-onnx", action="store_true", default=True,
                   help="Export ONNX after training (default True)")
    p.add_argument("--export-coreml", action="store_true", default=False,
                   help="Also export CoreML")
    p.add_argument("--no-megadetector", action="store_true",
                   help="Skip MegaDetector; use whole-image bbox fallback")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    datasets_root = (args.out or repo_root.parent / "datasets").resolve()
    mkdir(datasets_root)

    device = args.device or detect_device()
    log("quickstart", f"Device: {device}")

    species_list: list[str] = (
        list(SPECIES_CONFIG.keys()) if "all" in args.species else args.species
    )
    bad = [s for s in species_list if s not in SPECIES_CONFIG]
    if bad:
        print(f"[error] Unknown species: {bad}", file=sys.stderr)
        sys.exit(1)

    multi_class = len(species_list) > 1
    dataset_name = "north_american_wildlife" if multi_class else species_list[0]
    out_root = datasets_root / dataset_name
    mkdir(out_root)

    # ── 1. Download + bbox per species ────────────────────────────────────────
    all_pairs: list[tuple[Path, Path]] = []
    for sp in species_list:
        raw_dir     = datasets_root / sp / "raw"
        label_cache = datasets_root / sp / "labels_cache"
        md_json     = datasets_root / sp / "megadetector_results.json"
        class_id    = SPECIES_CONFIG[sp]["class_id"] if multi_class else 0

        # Download
        if not args.resume or not raw_dir.exists() or not any(raw_dir.glob("*.jpg")):
            downloaded = download_inaturalist(sp, raw_dir, args.max_images)
            if downloaded == 0:
                log("quickstart", f"No images downloaded for {sp} — skipping.")
                continue
        else:
            existing = sum(1 for _ in raw_dir.glob("*.jpg"))
            log("quickstart", f"{sp}: found {existing} cached images, skipping download.")

        # Bbox annotation
        if args.no_megadetector:
            log("quickstart", f"{sp}: using whole-image bbox fallback.")
            pairs = build_labels_whole_image(raw_dir, label_cache, class_id)
        else:
            if not args.resume or not md_json.exists():
                run_megadetector(raw_dir, md_json, device)
            else:
                log("megadetector", f"{sp}: cached results found, skipping.")

            if md_json.exists():
                pairs = build_labels_from_megadetector(raw_dir, md_json, label_cache, class_id)
                log("quickstart", f"{sp}: {len(pairs)} images with animal detections")
                if len(pairs) < 10:
                    log("quickstart", f"{sp}: too few MegaDetector hits; "
                        "falling back to whole-image bbox.")
                    pairs = build_labels_whole_image(raw_dir, label_cache, class_id)
            else:
                pairs = build_labels_whole_image(raw_dir, label_cache, class_id)

        all_pairs.extend(pairs)

    if not all_pairs:
        log("error", "No labelled samples — cannot train. Check network access.")
        sys.exit(1)

    log("quickstart", f"Total labelled samples: {len(all_pairs)}")

    # ── 2. Split into train/val/test ──────────────────────────────────────────
    if not args.resume or not (out_root / "images" / "train").exists():
        split_and_copy(all_pairs, out_root)
    else:
        log("quickstart", "Dataset split already exists, skipping copy.")

    # ── 3. Write YAML ─────────────────────────────────────────────────────────
    config_path = out_root / f"{dataset_name}.yaml"
    write_yaml_config(out_root, species_list, config_path)

    # Also update the training/configs copy
    training_cfg = repo_root / "training" / "configs" / f"{dataset_name}.yaml"
    # Rewrite path to point to absolute dataset location so training finds it
    lines = config_path.read_text().splitlines()
    abs_lines = [f"path: {out_root}" if l.startswith("path:") else l for l in lines]
    training_cfg.write_text("\n".join(abs_lines) + "\n")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    run_name   = dataset_name
    project_dir = repo_root / "training" / "runs" / "train"
    mkdir(project_dir)

    best_pt = train(config_path, args.base, args.epochs, device, run_name, project_dir)

    if not best_pt.exists():
        log("error", "Training did not produce best.pt — check logs.")
        sys.exit(1)

    # Copy best.pt to models/ directory
    models_dir = mkdir(repo_root / "models" / dataset_name)
    shutil.copy2(best_pt, models_dir / "best.pt")
    log("quickstart", f"Weights copied → {models_dir / 'best.pt'}")

    # ── 5. Export ─────────────────────────────────────────────────────────────
    if args.export_onnx:
        export_model(best_pt, "onnx", device)
        onnx_src = best_pt.with_suffix(".onnx")
        if onnx_src.exists():
            shutil.copy2(onnx_src, models_dir / "best.onnx")
            log("quickstart", f"ONNX → {models_dir / 'best.onnx'}")

    if args.export_coreml:
        export_model(best_pt, "coreml", device)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QUICKSTART COMPLETE")
    print("=" * 60)
    print(f"  PyTorch weights: {models_dir / 'best.pt'}")
    if args.export_onnx:
        print(f"  ONNX weights:    {models_dir / 'best.onnx'}")
    print(f"  Training logs:   {project_dir / run_name}")
    print()
    print("Load in Python:")
    print("  from ultralytics import YOLO")
    print(f"  model = YOLO('{models_dir / 'best.pt'}')")
    print("  results = model.predict('image.jpg', conf=0.25)")
    print("=" * 60)


if __name__ == "__main__":
    main()
