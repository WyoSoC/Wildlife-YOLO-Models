# Model Card — Bighorn Sheep (*Ovis canadensis*)

Single-class YOLO11s detection model for bighorn sheep.
Class imbalance is significant: bighorn make up only 0.60 % of NACTI images.

## Training Data

| Source | Images | Notes |
|---|---|---|
| NACTI (LILA BC) | ~23,800 | 0.60% of dataset; oversampled 3× to address imbalance |
| WCS Camera Traps | supplementary | |

## Training Details

| Property | Value |
|---|---|
| Architecture | YOLO11s |
| Input size | 640 × 640 |
| Epochs | 100 |
| Oversampling | 3× minority class |
| Device | CUDA |

## Performance

| Metric | Value |
|---|---|
| mAP50 | TBD |
| mAP50-95 | TBD |
| Precision | TBD |
| Recall | TBD |

## Known Limitations

- Limited training data may reduce generalisation to novel camera deployments.
- Mountain goats and domestic sheep can produce false positives.

## HuggingFace

<https://huggingface.co/WyoSoC/wildlife-bighorn-sheep>
