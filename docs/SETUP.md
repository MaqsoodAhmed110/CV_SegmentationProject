# Setup & Installation Guide

## Prerequisites
- Python 3.8+
- Google Colab account (for GPU training and inference)
- Git

## 1. Local Environment Setup
Clone the repository and install the basic dependencies required for the calibration phase.

```bash
git clone <repository_url>
cd cv_project

# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 2. Camera Calibration
To run the intrinsic camera calibration:
1. Place your 20+ checkerboard images in `calibration/images/`.
2. Run the script:
```bash
python calibration/calibrate.py
```
This will output `calibration/calibration_params.pkl` which is strictly required for the un-distortion process during measurement.

## 3. Deep Learning Training (Google Colab)
Detectron2 (by Meta AI) is utilized for instance segmentation. Due to local Windows C++ build constraints, training is performed on Google Colab.
1. Upload your exported `dataset.zip` and `models/segmetation.ipynb` to Colab.
2. Execute the commands outlined in `models/COLAB_INSTRUCTIONS.md`.
3. Download the resulting `model_final.pth` into your local `models/` directory.

## 4. Measurement Inference
To execute the end-to-end pixel-to-millimetre measurement pipeline:

**In Google Colab (Recommended for Windows Users):**
1. Upload `calibration/calibration_params.pkl`, `models/model_final.pth`, and zip your `measurement/` folder to Colab.
2. Unzip the measurement folder and run the pipeline:
```bash
python measurement/measure.py
```

The annotated output images with the calculated millimetre dimensions will be generated in the `measurement/outputs/` folder.
