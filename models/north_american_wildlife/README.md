# Model Card — North American Wildlife (multi-species)

Seven-class YOLO11m detection model for common wildlife species in
the Greater Yellowstone Ecosystem and western U.S.

## Classes

| ID | Species | Latin name |
|---|---|---|
| 0 | Golden Eagle | *Aquila chrysaetos* |
| 1 | Pronghorn | *Antilocapra americana* |
| 2 | Bighorn Sheep | *Ovis canadensis* |
| 3 | American Bison | *Bison bison* |
| 4 | Mule Deer | *Odocoileus hemionus* |
| 5 | Elk | *Cervus canadensis* |
| 6 | Coyote | *Canis latrans* |

## Training Data

Combined from NACTI, WCS Camera Traps, and iNaturalist.
Minority classes (bighorn sheep, golden eagle) are oversampled.

## Training Details

| Property | Value |
|---|---|
| Architecture | YOLO11m |
| Input size | 640 × 640 |
| Epochs | 150 |
| Augmentation | mosaic=1.0, mixup=0.1, degrees=10°, scale=0.5 |
| Device | CUDA (A100) |

## Performance

| Class | mAP50 | mAP50-95 |
|---|---|---|
| golden_eagle | TBD | TBD |
| pronghorn | TBD | TBD |
| bighorn_sheep | TBD | TBD |
| bison | TBD | TBD |
| mule_deer | TBD | TBD |
| elk | TBD | TBD |
| coyote | TBD | TBD |
| **mean** | **TBD** | **TBD** |

## Known Limitations

- Elk vs. mule deer confusion is the most common error; very similar silhouette.
- Low-light / IR images reduce species-level discrimination (shapes look similar).

## HuggingFace

<https://huggingface.co/UWyo/wildlife-north-american>
