# Deep Learning for Binary Classification of Breast Cancer in Ultrasound Images

Capstone project (ICML-style paper format) — co-authored with Cecil Swanzy and Mariana Resendiz Chavez.

## Problem

Classify breast ultrasound images as malignant vs. non-malignant (collapsing the original Normal/Benign/Malignant labels into a binary task). Dataset: 780 images from 600 patients (2018), split 70/20/10, with significant class imbalance (570 non-malignant vs. 210 malignant).

## Approach

Built and benchmarked 5 architectures:
- Custom CNN (baseline)
- Enhanced CNN
- VGG16 (transfer learning)
- ResNet50 (transfer learning)
- Voting ensemble of the above

**The core challenge:** the baseline CNN reached 75.6% accuracy but only **9.5% sensitivity** on malignant cases — a dangerous failure mode in a medical-screening context, since accuracy alone masked the model missing most actual cancer cases.

**Fixes applied:**
- Weighted/focal loss functions to penalize missed malignant cases more heavily
- Aggressive data augmentation targeted at the minority (malignant) class
- Transfer learning from ImageNet-pretrained VGG16/ResNet50
- Ensembling across all architectures

## Results

| Model | Sensitivity | Specificity | AUC |
|---|---|---|---|
| Baseline CNN | 9.5% | high | low |
| VGG16 (transfer learning) | 81.0% | 56.1% | — |
| Ensemble | **95.2%** | — | **0.767 (best)** |

The ensemble model achieved the best AUC and near-perfect sensitivity; VGG16 offered a more balanced sensitivity/specificity trade-off depending on the clinical use case.

## Skills demonstrated
Medical imaging · CNN architecture design · transfer learning · class-imbalance handling (weighted loss, focal loss, augmentation) · ensemble methods · rigorous evaluation (precision/recall/F1/AUC-ROC)

## Files
- `report.pdf` — full write-up
- Code files — see this folder
