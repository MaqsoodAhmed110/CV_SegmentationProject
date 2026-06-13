# COLAB TRAINING INSTRUCTIONS

Since we are strictly forbidden from using Ultralytics YOLO models, we will use **Detectron2** (by Meta AI Research). It is an industry-standard, professional-grade framework that uses **Mask R-CNN** (ResNet-50 FPN) for instance segmentation.

This satisfies the assessment's "Model Selection" requirement perfectly!

### Step 1: Prepare Google Colab
1. Go to Google Colab and create a New Notebook.
2. Go to **Runtime > Change runtime type** and select **T4 GPU** (or any GPU).

### Step 2: Upload Your Dataset and Code
1. Upload your exported `dataset.zip` (from Roboflow) to the Colab files section (left sidebar).
2. Upload the `colab_training.py` file (located in your local `models/` folder) to the Colab files section.

### Step 3: Run the Setup and Training
In your Colab notebook, create a code cell and paste the following commands to unzip your data, install Detectron2, and start training:

```python
# 1. Unzip the dataset (assuming the zip is named dataset.zip)
!unzip -q dataset.zip -d dataset/

# 2. Install Dependencies for Detectron2
!python -m pip install pyyaml==5.1
import sys, os, distutils.core
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 3. Run the training script!
!python colab_training.py
```

### Step 4: What to do after training completes
Once training finishes:
1. It will output a folder called `output/` in Colab.
2. Download the file `output/model_final.pth` (these are your trained weights).
3. Download the `output/metrics.json` file.
4. Move both of these files into your local `models/` directory on your computer.
5. Send me the final printed evaluation metrics (mAP, IoU) so I can draft your `TRAINING_REPORT.md`!
