# Dataset Card

## Object Selection
- **Object**: Tissue Box
- **Ground Truth Width**: 101.6 mm
- **Ground Truth Height**: 50.8 mm
- **Reasoning**: Tissue boxes have rigid, well-defined straight edges and predictable geometries, making them an excellent choice for bounding box/mask measurements and ensuring accuracy.

## Collection Strategy
Collected 70+ diverse images of a tissue box from varied angles, lighting conditions, and backgrounds using a mobile phone camera. The dataset was split into Train (70%), Validation (20%), and Test (10%) sets.

## Labelling Tool
Images were meticulously labelled using **Roboflow** with the polygon tool to create precise instance segmentation masks. The dataset was exported in the standard **COCO Segmentation** format.

## Class Distribution
- `tissue_box`: 100% (Single-class instance segmentation)
