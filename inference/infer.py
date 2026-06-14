import cv2
import pickle
import numpy as np
import torch

try:
    import gradio as gr
except ImportError:
    print("Please install gradio: pip install gradio")

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
except ImportError:
    print("Detectron2 not installed. Inference will be skipped.")

# --- Hardcoded Colab Paths ---
CALIBRATION_FILE = '/content/calibration_params.pkl'
WEIGHTS_FILE = '/content/model_final.pth'

# Load Calibration
mtx, dist = None, None
try:
    with open(CALIBRATION_FILE, 'rb') as f:
        calib = pickle.load(f)
    mtx, dist = calib['camera_matrix'], calib['dist_coeff']
except Exception as e:
    print(f"Warning: Calibration file not found at {CALIBRATION_FILE}. Un-distortion skipped.")

# Setup Predictor
def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = WEIGHTS_FILE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

try:
    predictor = setup_predictor()
except Exception as e:
    predictor = None
    print(f"Warning: Model weights not found at {WEIGHTS_FILE}.")

def process_image(img):
    if img is None:
        return None
    
    # Gradio passes images in RGB format. OpenCV uses BGR.
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    # 1. Un-distort the image
    if mtx is not None and dist is not None:
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        img_bgr = cv2.undistort(img_bgr, mtx, dist, None, newcameramtx)
        
    output_img = img_bgr.copy()
    
    # 2. Run Inference
    if predictor:
        outputs = predictor(img_bgr)
        instances = outputs["instances"]
        
        if len(instances) > 0:
            masks = instances.pred_masks.to("cpu").numpy()
            mask = (masks[0] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                # Draw the mask outline in Red
                cv2.drawContours(output_img, [c], -1, (0, 0, 255), 3)
                
    # Convert back to RGB for Gradio frontend
    return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

# Build the Gradio UI
app = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Upload Raw Image"),
    outputs=gr.Image(label="Un-distorted & Segmented Output"),
    title="XIS Assessment: End-to-End Inference",
    description="Upload a raw photo. The app will automatically un-distort it using your intrinsic calibration matrix and overlay the Colab-trained Mask R-CNN segmentation mask.",
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Launching Gradio App...")
    app.launch(share=True, debug=True)
