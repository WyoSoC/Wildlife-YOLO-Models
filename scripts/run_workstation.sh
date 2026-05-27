#!/usr/bin/env bash
# ============================================================
# Wildlife YOLO — Workstation Training Pipeline
#
# Runs the full pipeline on a CUDA workstation:
#   1. Install Python dependencies
#   2. Download 300 images/species from iNaturalist (skips existing)
#   3. Run MegaDetector v5a for bounding-box generation
#   4. Build YOLO-format dataset (train/val/test split)
#   5. Train 25-class north_american_wildlife model  (yolo11s, 100 epochs)
#   6. Train dedicated golden_eagle model             (yolo11s, 150 epochs)
#   7. Export both models to ONNX
#
# Usage:
#   chmod +x scripts/run_workstation.sh
#   ./scripts/run_workstation.sh
#
# Optional env overrides:
#   DEVICE=cuda        (default: auto-detect cuda > mps > cpu)
#   EPOCHS_MULTI=100   (25-class model epochs)
#   EPOCHS_EAGLE=150   (golden_eagle model epochs)
#   BASE_MODEL=yolo11s.pt
#   DATASETS_DIR=/path/to/datasets   (default: ../datasets relative to repo)
#
# Requirements: Python 3.10+, pip, git, internet access
# ============================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env if present (sets HF_TOKEN, HF_ORG, etc.)
if [ -f "$REPO_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_DIR/.env"
    set +a
    echo "[env] Loaded $REPO_DIR/.env"
fi
DATASETS_DIR="${DATASETS_DIR:-$(dirname "$REPO_DIR")/datasets}"
DEVICE="${DEVICE:-auto}"
EPOCHS_MULTI="${EPOCHS_MULTI:-100}"
EPOCHS_EAGLE="${EPOCHS_EAGLE:-150}"
BASE_MODEL="${BASE_MODEL:-yolo11s.pt}"
LOG_DIR="$REPO_DIR/training/logs"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "============================================================"
echo "Wildlife YOLO Workstation Pipeline"
echo "  Repo:     $REPO_DIR"
echo "  Datasets: $DATASETS_DIR"
echo "  Device:   $DEVICE"
echo "  Base:     $BASE_MODEL"
echo "  Started:  $(date)"
echo "============================================================"

# ── 1. Dependencies ──────────────────────────────────────────────────────────
echo ""
echo "[setup] Installing Python dependencies …"
pip install -q ultralytics torch torchvision \
    huggingface_hub datasets pycocotools \
    Pillow opencv-python-headless numpy \
    pandas matplotlib seaborn tqdm requests

# MegaDetector (idempotent)
pip install -q "git+https://github.com/agentmorris/MegaDetector.git" \
    2>/dev/null || echo "[setup] MegaDetector already installed."

echo "[setup] Dependencies ready."

# ── 2. Download images (skips existing files) ─────────────────────────────────
echo ""
echo "[download] Fetching 300 images/species from iNaturalist GBIF …"
python3 - << PYEOF
import requests, time
from pathlib import Path

GBIF_API  = "https://api.gbif.org/v1/occurrence/search"
INAT_KEY  = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
GOAL      = 300

SPECIES = {
    "golden_eagle":    2480506,
    "pronghorn":       2440902,
    "bighorn_sheep":   2441119,
    "bison":           2441176,
    "mule_deer":       2440974,
    "elk":             8600904,
    "coyote":          5219153,
    "grizzly_bear":    6163845,
    "gray_wolf":       5219173,
    "moose":           2440940,
    "pika":            2436982,
    "swift_fox":       5219290,   # only ~200 available on iNat
    "mountain_lion":   2435099,
    "river_otter":     2433727,
    "black_bear":      2433407,
    "bald_eagle":      2480446,
    "red_tailed_hawk": 2480542,
    "osprey":          2480726,
    "sage_grouse":     5959240,
    "trumpeter_swan":  2498345,
    "beaver":          2439838,
    "raven":           2482492,
    "prairie_dog":     2437232,
    "badger":          2434102,
    "bobcat":          2435246,
}

BASE = Path("${DATASETS_DIR}")

for species, taxon in SPECIES.items():
    raw_dir = BASE / species / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in raw_dir.glob("*.jpg")}
    need = max(0, GOAL - len(existing))
    print(f"[{species}] have {len(existing)}, need {need} more", flush=True)
    if need == 0:
        continue

    new_count, offset, empty_streak = 0, 0, 0
    while new_count < need:
        try:
            r = requests.get(GBIF_API, params={
                "taxonKey": taxon, "datasetKey": INAT_KEY,
                "mediaType": "StillImage", "limit": 100, "offset": offset,
            }, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  GBIF error: {e} — retrying", flush=True)
            time.sleep(5); continue

        data = r.json()
        results = data.get("results", [])
        if not results or data.get("endOfRecords"):
            break

        batch_new = 0
        for occ in results:
            if new_count >= need: break
            key = str(occ["key"])
            if key in existing: continue
            for media in occ.get("media", []):
                url = media.get("identifier")
                if not url: continue
                dest = raw_dir / f"{key}.jpg"
                try:
                    img = requests.get(url, timeout=15)
                    if img.status_code == 200:
                        dest.write_bytes(img.content)
                        existing.add(key)
                        new_count += 1
                        batch_new += 1
                        if new_count % 50 == 0:
                            print(f"  {species}: {new_count}/{need}", flush=True)
                except Exception:
                    pass
                break

        offset += 100
        empty_streak = 0 if batch_new else empty_streak + 1
        if empty_streak >= 3:
            break

    total = len(list(raw_dir.glob("*.jpg")))
    print(f"  {species}: done — {total} total images", flush=True)

print("Download complete.")
PYEOF

# ── 3+4+5. 25-class model ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STEP A — 25-class north_american_wildlife (${BASE_MODEL}, ${EPOCHS_MULTI} epochs)"
echo "============================================================"

DEVICE_ARG="$DEVICE"
if [ "$DEVICE_ARG" = "auto" ]; then
    DEVICE_ARG=$(python3 -c "
import torch
if torch.cuda.is_available(): print('cuda')
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): print('mps')
else: print('cpu')
")
fi
echo "[train] Using device: $DEVICE_ARG"

python3 scripts/quickstart.py \
    --species all \
    --resume \
    --base "$BASE_MODEL" \
    --epochs "$EPOCHS_MULTI" \
    --device "$DEVICE_ARG" \
    --export-onnx \
    2>&1 | tee "$LOG_DIR/train_25class_$(date +%Y%m%d_%H%M%S).log"

# ── 6+7. Dedicated golden eagle model ────────────────────────────────────────
echo ""
echo "============================================================"
echo "STEP B — Dedicated golden_eagle model (${BASE_MODEL}, ${EPOCHS_EAGLE} epochs)"
echo "============================================================"
python3 scripts/quickstart.py \
    --species golden_eagle \
    --resume \
    --base "$BASE_MODEL" \
    --epochs "$EPOCHS_EAGLE" \
    --device "$DEVICE_ARG" \
    --export-onnx \
    --export-coreml \
    2>&1 | tee "$LOG_DIR/train_golden_eagle_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE — $(date)"
echo ""
echo "Outputs:"
echo "  models/north_american_wildlife/best.pt   (25-class)"
echo "  models/north_american_wildlife/best.onnx"
echo "  models/golden_eagle/best.pt              (dedicated)"
echo "  models/golden_eagle/best.onnx"
echo "  models/golden_eagle/best.mlpackage       (CoreML)"
echo "  training/logs/                           (training logs)"
echo "============================================================"
