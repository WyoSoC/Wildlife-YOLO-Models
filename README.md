# Wildlife YOLO Models

YOLO detection models for North American wildlife, built on
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/).
Model weights are hosted on HuggingFace; this repository contains
the training pipelines, dataset preparation scripts, and model cards.

**HuggingFace collection:** https://huggingface.co/WyoSoC

**Companion app:** [Wildlife PTZ Camera Tracker](https://github.com/WyoSoC/Wildlife_PTZ_Camera_Tracker)
— drop any `.pt` file from this repo into the app's `models/` folder to activate it.

---

## Available Models

| Model | Species | Base | Status | HuggingFace |
|---|---|---|---|---|
| `golden_eagle` | Golden Eagle (*Aquila chrysaetos*) | YOLO11s | 🔜 In training | — |
| `pronghorn` | Pronghorn (*Antilocapra americana*) | YOLO11s | 🔜 In training | — |
| `bighorn_sheep` | Bighorn Sheep (*Ovis canadensis*) | YOLO11s | 🔜 In training | — |
| `bison` | American Bison (*Bison bison*) | YOLO11s | 🔜 In training | — |
| `north_american_wildlife` | Multi-species (7 classes) | YOLO11m | 🔜 Planned | — |

### Multi-species classes (north_american_wildlife)

| ID | Species |
|---|---|
| 0 | Golden Eagle |
| 1 | Pronghorn |
| 2 | Bighorn Sheep |
| 3 | American Bison |
| 4 | Mule Deer |
| 5 | Elk |
| 6 | Coyote |

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
| [NACTI](https://lila.science/datasets/nacti/) (3.7M images) | Pronghorn ✓, Bighorn Sheep ✓, Deer, Elk, Bear | Sparse bbox (0.24%) + species labels | GCP / AWS / Azure |
| [WCS Camera Traps](https://lila.science/datasets/wcscameratraps/) (1.4M images) | 675 species, global | 375K bbox annotations | [HuggingFace](https://huggingface.co/datasets/society-ethics/lila_camera_traps) |
| [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps) (243K images) | Southwest US species | 66K bbox annotations | LILA BC |
| [iNaturalist](https://www.inaturalist.org/taxa/5074-Aquila-chrysaetos) | Golden Eagle 10K+ obs | Classification only (no bbox) | GBIF export |

### Annotation pipeline for sparse-bbox datasets

Most camera trap datasets have species labels but few bounding boxes.
We use **MegaDetector v5** to auto-generate animal bboxes on any image,
then combine with the dataset's species labels to create YOLO training data:

```
Raw images + species labels
        │
        ▼
MegaDetector → bounding boxes (animal / human / vehicle)
        │
        ▼
Filter: keep "animal" detections matching target species
        │
        ▼
Convert COCO → YOLO label format
        │
        ▼
Fine-tune YOLO11s/m
```

---

## Training a Model

### 1 — Install dependencies

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Download dataset

```bash
# NACTI from LILA BC (requires gsutil or aws cli; downloads ~400 GB subset)
python 01_download_dataset.py --species pronghorn --source nacti --out ../datasets/

# WCS Camera Traps via HuggingFace (lighter, no CLI tool needed)
python 01_download_dataset.py --species pronghorn --source wcs --out ../datasets/
```

### 3 — Prepare dataset (COCO → YOLO format + MegaDetector bbox)

```bash
python 02_prepare_dataset.py --species pronghorn --dataset ../datasets/nacti/
```

### 4 — Train

```bash
python 03_train.py --config configs/pronghorn.yaml --base yolo11s.pt --device cuda
# On Apple Silicon:  --device mps
# On Jetson Orin:    --device cuda --half
```

### 5 — Evaluate

```bash
python 04_evaluate.py --weights runs/train/pronghorn/weights/best.pt \
                      --data configs/pronghorn.yaml
```

### 6 — Export

```bash
# ONNX (universal — works on any device)
python 05_export.py --weights runs/train/pronghorn/weights/best.pt --format onnx

# TensorRT engine (Jetson Orin only — run this ON the Jetson)
python 05_export.py --weights runs/train/pronghorn/weights/best.pt --format engine --half
```

### 7 — Upload to HuggingFace

```bash
huggingface-cli login
python 06_upload_to_hf.py --weights runs/train/pronghorn/weights/best.pt \
                           --model-id WyoSoC/wildlife-pronghorn \
                           --species pronghorn
```

---

## Hardware Requirements

| Task | Minimum | Recommended |
|---|---|---|
| MegaDetector bbox generation | 8 GB RAM, CPU | NVIDIA GPU (3–10× faster) |
| YOLO11s fine-tuning | RTX 3070 / Jetson AGX | RTX 4090 / A100 |
| YOLO11m fine-tuning | RTX 3090 | A100 / H100 |
| Inference (edge) | Raspberry Pi 5 (3–5 fps) | Jetson Orin Nano (30 fps) |

---

## Repository Layout

```
.
├── models/                         # Model cards (one folder per species)
│   ├── golden_eagle/README.md
│   ├── pronghorn/README.md
│   ├── bighorn_sheep/README.md
│   ├── bison/README.md
│   └── north_american_wildlife/README.md
├── training/
│   ├── requirements.txt
│   ├── configs/                    # YOLO dataset YAML configs
│   │   ├── golden_eagle.yaml
│   │   ├── pronghorn.yaml
│   │   ├── bighorn_sheep.yaml
│   │   ├── bison.yaml
│   │   └── north_american_wildlife.yaml
│   ├── 01_download_dataset.py      # Download from LILA BC or HuggingFace
│   ├── 02_prepare_dataset.py       # COCO→YOLO + MegaDetector bbox
│   ├── 03_train.py                 # Fine-tune YOLO11
│   ├── 04_evaluate.py              # mAP, confusion matrix, speed benchmark
│   ├── 05_export.py                # ONNX / TensorRT / CoreML export
│   └── 06_upload_to_hf.py          # Push weights to HuggingFace Hub
└── scripts/
    └── download_models.py          # End-user: download a model from HuggingFace
```

---

## Contributing

1. Fork the repo
2. Train a model on a new species or improve an existing one
3. Evaluate on a held-out test set and record metrics in the model card
4. Upload weights to the WyoSoC HuggingFace org and open a PR

Species priorities: **Mule Deer, Elk, Mountain Lion, Black Bear, Coyote, Moose**

---

## License

Training code: MIT.
Dataset licenses vary — see each source (LILA BC datasets are typically CC-BY or similar).
Model weights: CC-BY-4.0 (same as the underlying training data license).
