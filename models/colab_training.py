import os
import json
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def main():
    print("Registering COCO Datasets...")
    # Adjust the paths below if your roboflow zip extracts differently
    register_coco_instances("tissue_train", {}, "dataset/train/_annotations.coco.json", "dataset/train")
    register_coco_instances("tissue_val", {}, "dataset/valid/_annotations.coco.json", "dataset/valid")

    print("Configuring Mask R-CNN Model...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("tissue_train",)
    cfg.DATASETS.TEST = ("tissue_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Load pre-trained weights for transfer learning
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size
    cfg.SOLVER.BASE_LR = 0.0005   # Learning rate
    cfg.SOLVER.MAX_ITER = 600     # Epochs / Iterations
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only 1 class: tissue_box

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("Starting Training...")
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("Training Complete! Running Evaluation on Validation Set...")
    evaluator = COCOEvaluator("tissue_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "tissue_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)
    print("Evaluation Complete. Check the output/ folder for model_final.pth and metrics.json.")

if __name__ == "__main__":
    main()
