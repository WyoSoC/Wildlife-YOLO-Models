# Wildlife YOLO Models

YOLO detection models for North American wildlife, built on
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/).
Model weights are hosted on HuggingFace; this repository contains
the training pipelines, dataset preparation scripts, and model cards.

**HuggingFace collection:** https://huggingface.co/UWyo

**Companion app:** [Wildlife PTZ Camera Tracker](https://github.com/WyoSoC/Wildlife_PTZ_Camera_Tracker)
— drop any `.pt` file from this repo into the app's `models/` folder to activate it.

---

## Available Models

| Model | Classes | Base | Status | HuggingFace |
|---|---|---|---|---|
| `north_american_wildlife` | 25 species (see below) | YOLO11s | 🔜 In training | — |
| `golden_eagle` | Golden Eagle (*Aquila chrysaetos*) | YOLO11s | 🔜 In training | — |
| `pronghorn` | Pronghorn (*Antilocapra americana*) | YOLO11s | 🔜 Planned | — |
| `bighorn_sheep` | Bighorn Sheep (*Ovis canadensis*) | YOLO11s | 🔜 Planned | — |
| `bison` | American Bison (*Bison bison*) | YOLO11s | 🔜 Planned | — |

### Multi-species classes (`north_american_wildlife`)

| ID | Species | Latin name | Notes |
|---|---|---|---|
| 0 | Golden Eagle | *Aquila chrysaetos* | Primary research species — 1,200+ training images |
| 1 | Pronghorn | *Antilocapra americana* | |
| 2 | Bighorn Sheep | *Ovis canadensis* | |
| 3 | American Bison | *Bison bison* | |
| 4 | Mule Deer | *Odocoileus hemionus* | |
| 5 | Elk | *Cervus canadensis* | |
| 6 | Coyote | *Canis latrans* | |
| 7 | Grizzly Bear | *Ursus arctos horribilis* | |
| 8 | Gray Wolf | *Canis lupus* | |
| 9 | Moose | *Alces alces* | |
| 10 | Pika | *Ochotona princeps* | |
| 11 | Swift Fox | *Vulpes velox* | Limited training data (~200 images) |
| 12 | Mountain Lion | *Puma concolor* | Limited training data (~137 images) |
| 13 | River Otter | *Lontra canadensis* | |
| 14 | Black Bear | *Ursus americanus* | |
| 15 | Bald Eagle | *Haliaeetus leucocephalus* | |
| 16 | Red-tailed Hawk | *Buteo jamaicensis* | |
| 17 | Osprey | *Pandion haliaetus* | |
| 18 | Greater Sage-Grouse | *Centrocercus urophasianus* | |
| 19 | Trumpeter Swan | *Cygnus buccinator* | |
| 20 | Beaver | *Castor canadensis* | Limited training data (~64 images) |
| 21 | Common Raven | *Corvus corax* | |
| 22 | Black-tailed Prairie Dog | *Cynomys ludovicianus* | |
| 23 | American Badger | *Taxidea taxus* | |
| 24 | Bobcat | *Lynx rufus* | |

---

## Quick Start — Download & Use a Model

```bash
pip install huggingface_hub
python scripts/download_models.py --model golden_eagle --out models/
```

In the Wildlife PTZ Camera Tracker app, copy the downloaded `.pt` file into the
`models/` directory at the project root. The model will appear automatically in
the Camera & Config tab.

---

## Data Sources

| Dataset | Species covered | Annotations | Access |
|---|---|---|---|
| [NACTI](https://lila.science/datasets/nacti/) (3.7M images) | Pronghorn, Bighorn Sheep, Bison, Elk, Deer, Bear, Coyote | Sparse bbox + species labels | GCP / AWS / Azure |
| [WCS Camera Traps](https://lila.science/datasets/wcscameratraps/) (1.4M images) | 675 species, global | 375K bbox annotations | [HuggingFace](https://huggingface.co/datasets/society-ethics/lila_camera_traps) |
| [iNaturalist via GBIF](https://www.gbif.org/) | All 25 species | Classification only (no bbox) | Public API — no auth required |

All iNaturalist downloads use **GBIF backbone taxon keys** (not iNaturalist internal IDs).
See `SPECIES_CONFIG` in `training/01_download_dataset.py` for the full key mapping.

### Annotation pipeline for sparse-bbox datasets

```
Raw images + species labels
        │
        ▼
MegaDetector v5a → bounding boxes (animal / human / vehicle)
        │
        ▼
Filter: keep "animal" detections ≥ 0.15 confidence
        │
        ▼
Convert to YOLO label format (cx cy w h, normalised)
        │
        ▼
80/10/10 train/val/test split
        │
        ▼
Fine-tune YOLO11s/m
```

---

## Training a Model

### Option A — Workstation (recommended for full training runs)

Clone the repo on your workstation and run the single-command pipeline:

```bash
git clone https://github.com/WyoSoC/Wildlife-YOLO-Models.git
cd Wildlife-YOLO-Models
chmod +x scripts/run_workstation.sh
./scripts/run_workstation.sh
```

The script installs all dependencies, downloads images from iNaturalist, runs
MegaDetector for bbox generation, trains both models, and exports ONNX weights.
Safe to interrupt and re-run — already-downloaded images and MegaDetector results
are cached and skipped automatically.

**Environment variables (optional overrides):**

```bash
DEVICE=cuda             # auto-detects cuda > mps > cpu if not set
EPOCHS_MULTI=100        # epochs for the 25-class model (default 100)
EPOCHS_EAGLE=150        # epochs for the dedicated golden eagle model (default 150)
BASE_MODEL=yolo11m.pt   # upgrade to medium if VRAM allows (≥16 GB recommended)
DATASETS_DIR=/data/wildlife/datasets   # store datasets on a separate disk
```

**Expected runtimes on a CUDA workstation (RTX 4090 / A100):**

| Step | Time |
|---|---|
| Image download (25 × 300) | 20–40 min |
| MegaDetector on ~7,200 images | 15–30 min |
| 25-class training, 100 epochs | 1–3 hours |
| Golden eagle training, 150 epochs | 30–60 min |

### Option B — Manual step-by-step

#### 1 — Install dependencies

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/agentmorris/MegaDetector.git"
```

#### 2 — Download dataset

```bash
# iNaturalist via GBIF (no auth required, all 25 species):
python 01_download_dataset.py --species all --source inaturalist --out ../datasets/

# NACTI from LILA BC (requires gsutil; downloads ~400 GB subset):
python 01_download_dataset.py --species pronghorn --source nacti --out ../datasets/

# WCS Camera Traps via HuggingFace:
python 01_download_dataset.py --species pronghorn --source wcs --out ../datasets/
```

#### 3 — Prepare dataset (MegaDetector bbox + COCO → YOLO format)

```bash
# Single species:
python 02_prepare_dataset.py --species golden_eagle --dataset ../datasets/ --megadetector

# All 25 species (multi-class dataset):
python 02_prepare_dataset.py --species all --dataset ../datasets/ --megadetector --balance
```

#### 4 — Train

```bash
# 25-class model:
python 03_train.py --config configs/north_american_wildlife.yaml \
                   --base yolo11s.pt --device cuda --epochs 100

# Dedicated single-species model:
python 03_train.py --config configs/golden_eagle.yaml \
                   --base yolo11s.pt --device cuda --epochs 150

# Apple Silicon:
python 03_train.py --config configs/golden_eagle.yaml --base yolo11s.pt --device mps

# Resume interrupted training:
python 03_train.py --config configs/golden_eagle.yaml \
                   --resume runs/train/golden_eagle/weights/last.pt
```

#### 5 — Evaluate

```bash
python 04_evaluate.py --weights runs/train/golden_eagle/weights/best.pt \
                      --data configs/golden_eagle.yaml

# Sweep confidence thresholds to find the optimal operating point:
python 04_evaluate.py --weights runs/train/golden_eagle/weights/best.pt \
                      --data configs/golden_eagle.yaml --conf-sweep
```

#### 6 — Export

```bash
# ONNX (universal — CPU, CUDA, Jetson, ONNX Runtime):
python 05_export.py --weights runs/train/golden_eagle/weights/best.pt --format onnx

# CoreML (macOS / iOS):
python 05_export.py --weights runs/train/golden_eagle/weights/best.pt --format coreml

# TensorRT engine (run ON the target Jetson):
python 05_export.py --weights runs/train/golden_eagle/weights/best.pt \
                    --format engine --half
```

#### 7 — Upload to HuggingFace

```bash
huggingface-cli login
python 06_upload_to_hf.py --weights runs/train/golden_eagle/weights/best.pt \
                           --model-id UWyo/wildlife-golden-eagle \
                           --species golden_eagle
```

### Quick local test (single command)

For a fast smoke test on any machine with internet access:

```bash
python scripts/quickstart.py --species golden_eagle pronghorn \
                              --max-images 150 --epochs 30
```

---

## Hardware Requirements

| Task | Minimum | Recommended |
|---|---|---|
| MegaDetector bbox generation | 8 GB RAM, CPU | NVIDIA GPU (10–30× faster than CPU) |
| YOLO11s fine-tuning (25 classes) | RTX 3080 / Jetson AGX | RTX 4090 / A100 |
| YOLO11m fine-tuning | RTX 3090 | A100 / H100 |
| Inference (edge) | Raspberry Pi 5 (3–5 fps) | Jetson Orin Nano (30 fps) |

---

## Repository Layout

```
.
├── models/                         # Model cards and trained weights (one folder per species)
│   ├── golden_eagle/
│   │   ├── README.md
│   │   ├── best.pt                 # PyTorch weights (after training)
│   │   └── best.onnx               # ONNX export
│   ├── north_american_wildlife/    # 25-class model
│   ├── pronghorn/
│   ├── bighorn_sheep/
│   ├── bison/
│   └── …
├── training/
│   ├── requirements.txt
│   ├── configs/                    # YOLO dataset YAML configs (one per species + multi)
│   │   ├── north_american_wildlife.yaml   # 25-class
│   │   ├── golden_eagle.yaml
│   │   ├── pronghorn.yaml
│   │   └── …                      # one per species
│   ├── 01_download_dataset.py      # Download from LILA BC / HuggingFace / iNaturalist
│   ├── 02_prepare_dataset.py       # MegaDetector bbox + COCO→YOLO format
│   ├── 03_train.py                 # Fine-tune YOLO11
│   ├── 04_evaluate.py              # mAP, confusion matrix, speed benchmark
│   ├── 05_export.py                # ONNX / TensorRT / CoreML export
│   └── 06_upload_to_hf.py          # Push weights to HuggingFace Hub
└── scripts/
    ├── run_workstation.sh          # One-command full pipeline for workstations
    ├── quickstart.py               # End-to-end pipeline (download→train→export)
    └── download_models.py          # End-user: download a trained model from HuggingFace
```

---

## Contributing

1. Fork the repo
2. Train a model on a new species or improve an existing one
3. Evaluate on a held-out test set and record metrics in the model card
4. Upload weights to the UWyo HuggingFace org and open a PR

**Data quality notes:**
- Always use **GBIF backbone taxon keys** when querying the GBIF API (not iNaturalist internal IDs — they differ and will pull wrong species)
- Visually inspect raw images before training; camera-trap datasets commonly contain footprints, scat, and habitat shots with no visible animal
- Species with < 150 training images (beaver, swift fox, mountain lion) will benefit most from additional data

---

## License

Training code: MIT.
Dataset licenses vary — see each source (LILA BC datasets are typically CC-BY or similar).
Model weights: CC-BY-4.0 (same as the underlying training data license).
