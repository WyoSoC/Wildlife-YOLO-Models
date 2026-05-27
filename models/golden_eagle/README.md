# Model Card — Golden Eagle (*Aquila chrysaetos*)

Single-class YOLO11s detection model for golden eagles, optimised for
PTZ camera tracking of soaring and perched birds.

## Training Data

| Source | Images | Notes |
|---|---|---|
| iNaturalist / GBIF | 10,000+ | Research-grade observations; classification only — bboxes from MegaDetector |
| eBird / Macaulay Library | supplementary | Flight images; important for PTZ tracking at distance |

## Training Details

| Property | Value |
|---|---|
| Architecture | YOLO11s |
| Input size | 640 × 640 |
| Epochs | 100 |
| Augmentation | degrees=15°, scale=0.5, fliplr=0.5, mosaic=1.0 |
| Device | CUDA (A100) |

## Performance

| Metric | Value |
|---|---|
| mAP50 | TBD |
| mAP50-95 | TBD |
| Precision | TBD |
| Recall | TBD |
| Inference (Jetson Orin Nano, FP16) | TBD ms/img |

## Known Limitations

- Performance degrades against sky backgrounds at distances > 500 m.
- Juvenile golden eagles (brown plumage, no gold nape) may be missed more often.
- Very small birds in wide-angle PTZ shots (< 10 px) have low recall.

## Usage

```python
from ultralytics import YOLO
model = YOLO("WyoSoC/wildlife-golden-eagle")
results = model.predict("flight.jpg", conf=0.25)
```

Or download via [scripts/download_models.py](../../scripts/download_models.py).

## HuggingFace

<https://huggingface.co/WyoSoC/wildlife-golden-eagle>
