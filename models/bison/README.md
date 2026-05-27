# Model Card — American Bison (*Bison bison*)

Single-class YOLO11s detection model for American bison,
trained on NACTI camera-trap images supplemented by iNaturalist observations.

## Training Data

| Source | Images | Notes |
|---|---|---|
| NACTI (LILA BC) | ~97,000 | 2.45% of dataset |
| iNaturalist / GBIF | supplementary | Research-grade observations; bboxes from MegaDetector |

## Training Details

| Property | Value |
|---|---|
| Architecture | YOLO11s |
| Input size | 640 × 640 |
| Epochs | 100 |
| Device | CUDA |

## Performance

| Metric | Value |
|---|---|
| mAP50 | TBD |
| mAP50-95 | TBD |
| Precision | TBD |
| Recall | TBD |

## Known Limitations

- Calves at distance (< 15 px height) have lower recall.
- Herd scenes with heavy occlusion: bounding boxes may merge individuals.

## HuggingFace

<https://huggingface.co/WyoSoC/wildlife-bison>
