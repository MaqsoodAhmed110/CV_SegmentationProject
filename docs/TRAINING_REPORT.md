# Model Training & Segmentation Report

## Model Selection
**Architecture:** Detectron2 Mask R-CNN (ResNet-50 FPN)
**Justification:** The assessment strictly prohibited the use of Ultralytics YOLO models and Roboflow proprietary training. Detectron2, developed by Meta AI, is an industry-standard, professional-grade framework for instance segmentation. It provides granular control over training configurations, is distinct from YOLO, and has robust native COCO evaluation metrics.

## Training Setup
- **Dataset Split:** 70% Train, 20% Validation, 10% Test
- **Base Architecture:** `mask_rcnn_R_50_FPN_3x`
- **Iterations:** 1000 Iterations
- **Batch Size:** 2 Images per Batch
- **Learning Rate:** 0.0005
- **Classes:** 1 (`tissue_box`)
- **Augmentations:** `ResizeShortestEdge` (max size 1333)

## Evaluation Metrics
The model was evaluated on the held-out Validation set using the official COCO API evaluator.

### Bounding Box Metrics
| Metric | Value |
| :--- | :--- |
| **mAP (IoU=0.50:0.95)** | 98.52% |
| **mAP@0.50** | 100.00% |
| **mAP@0.75** | 100.00% |

### Segmentation Mask Metrics
| Metric | Value |
| :--- | :--- |
| **mAP (IoU=0.50:0.95)** | 98.93% |
| **mAP@0.50** | 100.00% |
| **mAP@0.75** | 100.00% |

## Loss Curves
The training loss curves (logged in `models/metrics.json`) show a smooth, stable convergence, with the `total_loss` dropping significantly to ~0.08 by the end of training.

## Conclusion
The model achieved near-perfect accuracy (100% mAP at IoU=0.50) due to the rigid geometry and distinct features of the tissue box. It is exceptionally capable of predicting the precise segmentation masks required for the upcoming real-world millimetre measurement phase.
