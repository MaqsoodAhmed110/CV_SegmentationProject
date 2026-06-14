import cv2
import numpy as np
import glob
import os
import pickle
import math

# Try to import detectron2, but allow script to run partially if not installed (e.g. if run locally without Colab)
try:
    import torch
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    HAS_DETECTRON = True
except ImportError:
    HAS_DETECTRON = False
    print("WARNING: Detectron2 not found. The script will only calculate pixels_per_mm but cannot run the neural network.")
    print("Please run this script in Google Colab if you cannot install Detectron2 on Windows.")

# --- Configuration ---
# Hardcoded for Google Colab environment based on user upload locations
CALIBRATION_FILE = '/content/calibration_params.pkl'
IMAGES_DIR = '/content/measurement/images/*.jpeg'
OUTPUT_DIR = '/content/measurement/outputs'

# Ground Truth dimensions of the Tissue Box
GT_WIDTH_MM = 101.6
GT_HEIGHT_MM = 50.8

# Checkerboard Details (From user's image)
CHECKERBOARD = (6, 8)  # Inner corners
SQUARE_SIZE_MM = 20.0  # Physical size of one square

def get_pixels_per_mm(corners, rows, cols, square_mm):
    """Calculates the average pixels per millimeter using the checkerboard grid."""
    corners = corners.reshape(-1, 2)
    distances = []
    
    # Calculate horizontal distances between adjacent corners
    for r in range(rows):
        for c in range(cols - 1):
            idx1 = r * cols + c
            idx2 = r * cols + c + 1
            dist = np.linalg.norm(corners[idx1] - corners[idx2])
            distances.append(dist)
            
    # Calculate vertical distances between adjacent corners
    for r in range(rows - 1):
        for c in range(cols):
            idx1 = r * cols + c
            idx2 = (r + 1) * cols + c
            dist = np.linalg.norm(corners[idx1] - corners[idx2])
            distances.append(dist)
            
    avg_pixel_dist = np.mean(distances)
    return avg_pixel_dist / square_mm

def setup_predictor():
    if not HAS_DETECTRON: return None
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Changed to 2 based on the checkpoint mismatch error
    # Path to the weights trained on Colab
    cfg.MODEL.WEIGHTS = "/content/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # Set threshold
    # If no GPU available locally, force CPU
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Calibration Params
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: {CALIBRATION_FILE} not found.")
        return
        
    with open(CALIBRATION_FILE, 'rb') as f:
        calib = pickle.load(f)
    mtx = calib['camera_matrix']
    dist = calib['dist_coeff']

    images = glob.glob(IMAGES_DIR)
    if not images:
        images = glob.glob('/content/measurement/images/*.jpg') + glob.glob('/content/measurement/images/*.jpeg')
        
    if not images:
        print("No images found in /content/measurement/images/")
        return

    predictor = setup_predictor()
    
    width_errors = []
    height_errors = []
    
    for fname in images:
        img_name = os.path.basename(fname)
        print(f"\nProcessing {img_name}...")
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        
        # 1. UNDISTORT IMAGE
        # We use alpha=0 to keep the image size the same and avoid cropping the checkerboard
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        img_undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # 2. FIND PIXELS PER MM (via Checkerboard)
        gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        
        # Try multiple grid sizes in case the un-distortion cropped a row or warped it
        grids_to_try = [(6, 8), (8, 6), (5, 7), (7, 5), (6, 7), (7, 6), (6, 9), (9, 6)]
        ret = False
        for grid in grids_to_try:
            ret, corners = cv2.findChessboardCorners(gray, grid, flags)
            if ret:
                found_grid = grid
                break
                
        if not ret:
            # Fallback: Try on the RAW original image in case the calibration matrix destroyed the undistorted image
            gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for grid in grids_to_try:
                ret, corners = cv2.findChessboardCorners(gray_orig, grid, flags)
                if ret:
                    found_grid = grid
                    print(f"  [!] WARNING: Found board on RAW image. Un-distortion severely warped the image!")
                    break
                    
        if not ret:
            print(f"  [!] Could not detect ANY checkerboard corners. Skipping measurement.")
            continue
            
        pixels_per_mm = get_pixels_per_mm(corners, found_grid[0], found_grid[1], SQUARE_SIZE_MM)
        print(f"  > Scale found: {pixels_per_mm:.2f} pixels/mm")
        
        # Visualize checkerboard on output image
        output_img = img_undistorted.copy()
        cv2.drawChessboardCorners(output_img, CHECKERBOARD, corners, ret)

        # 3. RUN SEGMENTATION & MEASURE
        if HAS_DETECTRON and predictor is not None:
            outputs = predictor(img_undistorted)
            instances = outputs["instances"]
            
            if len(instances) == 0:
                print("  [!] No tissue box detected by the model.")
                continue
                
            # Get the mask of the most confident detection
            masks = instances.pred_masks.to("cpu").numpy()
            mask = (masks[0] * 255).astype(np.uint8)
            
            # Find contours of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                # Use MinAreaRect to handle rotated tissue boxes perfectly
                rect = cv2.minAreaRect(c) 
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                width_px, height_px = rect[1]
                
                # Align longest side to width (since 101.6 > 50.8)
                if width_px < height_px:
                    width_px, height_px = height_px, width_px
                    
                calc_width_mm = width_px / pixels_per_mm
                calc_height_mm = height_px / pixels_per_mm
                
                # Calculate Error
                w_err = abs(calc_width_mm - GT_WIDTH_MM)
                h_err = abs(calc_height_mm - GT_HEIGHT_MM)
                width_errors.append(w_err)
                height_errors.append(h_err)
                
                print(f"  > Measurement: {calc_width_mm:.1f}mm x {calc_height_mm:.1f}mm")
                print(f"  > Error: Width {w_err:.1f}mm, Height {h_err:.1f}mm")
                
                # Draw Box and Text
                cv2.drawContours(output_img, [box], 0, (0, 255, 0), 3)
                label = f"W: {calc_width_mm:.1f}mm, H: {calc_height_mm:.1f}mm"
                cv2.putText(output_img, label, (int(box[0][0]), int(box[0][1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Save output
        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, output_img)
        print(f"  > Saved annotated image to {out_path}")

    # Final Statistics
    if width_errors:
        mae_w = np.mean(width_errors)
        mae_h = np.mean(height_errors)
        mpe_w = (mae_w / GT_WIDTH_MM) * 100
        mpe_h = (mae_h / GT_HEIGHT_MM) * 100
        print("\n" + "="*40)
        print("FINAL ACCURACY REPORT")
        print(f"Mean Absolute Error (Width):  {mae_w:.2f} mm")
        print(f"Mean Absolute Error (Height): {mae_h:.2f} mm")
        print(f"Mean Percentage Error (W):    {mpe_w:.2f}%")
        print(f"Mean Percentage Error (H):    {mpe_h:.2f}%")
        print("="*40)

if __name__ == "__main__":
    main()
