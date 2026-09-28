# Deepfake Detection Model

## Model Name
`deepfake_detector_celebdfv2_efficientnetb4_96acc`

## Architecture
- **Backbone:** EfficientNet-B4
- **Head:** Linear(1792 → 512 → 1)
- **Framework:** PyTorch

## Dataset
- **Name:** Celeb-DF-v2
- **Training frames (subset):** 3993 real, 4000 fake
- **Test frames (subset):** 704 real, 6768 fake

## Performance (on 15% test subset)
| Metric | Score |
|--------|-------|
| Accuracy | 97.00% |
| AUC-ROC | 99.72% |
| F1 Score | 97.00% |
| Precision | 97.00% |
| Recall | 97.00% |
| Specificity | 97.00% |

## Training Details
| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch Size | 16 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss | BCEWithLogitsLoss |
| Image Size | 256x256 |

## Saved Formats
- `.pth` — Full model with metadata
- `_weights_only.pth` — Weights only
- `.onnx` — Universal format

## Date Trained
2026-05-10
