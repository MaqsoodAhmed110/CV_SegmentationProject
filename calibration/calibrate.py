import cv2
import numpy as np
import glob
import os
import pickle

def calibrate_camera():
    # --- Configuration ---
    # Change these values based on the checkerboard you print/use
    CHECKERBOARD = (6, 9) # Inner corners: (rows, columns)
    SQUARE_SIZE = 25.0    # Square size in mm (Measure your actual print)
    IMAGES_DIR = 'images/*'
    OUTPUT_FILE = 'calibration_params.pkl'

    # Termination criteria for corner sub-pixel optimization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    # Arrays to store object points and image points from all valid images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Gather all jpg and jpeg images
    images = glob.glob('images/*.jpg') + glob.glob('images/*.jpeg') + glob.glob('images/*.JPG') + glob.glob('images/*.JPEG')

    if not images:
        print(f"No .jpg or .jpeg images found.")
        print("Please place your checkerboard photos inside the 'calibration/images' folder.")
        return

    print(f"Found {len(images)} images. Searching for checkerboard corners...")
    valid_images_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_images_count += 1
        else:
            print(f"Could not find exact {CHECKERBOARD} corners in {fname}")

    if valid_images_count == 0:
        print(f"\nFailed to detect the checkerboard in any images.")
        print(f"Make sure your checkerboard actually has {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners.")
        return

    print(f"\nSuccessfully found corners in {valid_images_count} out of {len(images)} images.")
    print("Running intrinsic calibration...")

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    total_error = mean_error / len(objpoints)

    print("\n" + "="*40)
    print("--- CALIBRATION COMPLETE ---")
    print(f"Camera Matrix (Intrinsic Parameters):\n{mtx}")
    print(f"\nDistortion Coefficients:\n{dist}")
    print(f"\nTotal Reprojection Error: {total_error:.4f} pixels")
    print("="*40)

    # Save the parameters for inference/measurement later
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'camera_matrix': mtx, 'dist_coeff': dist}, f)

    print(f"\nCalibration parameters saved to {OUTPUT_FILE}")

    # Un-distort a sample image to show it works
    sample_img = cv2.imread(images[0])
    h, w = sample_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(sample_img, mtx, dist, None, newcameramtx)
    # Crop the image (optional, but removes curved black edges)
    x, y, w_roi, h_roi = roi
    dst = dst[y:y+h_roi, x:x+w_roi]
    
    output_sample_path = 'undistorted_sample.jpg'
    cv2.imwrite(output_sample_path, dst)
    print(f"Saved {output_sample_path} to verify calibration.")

if __name__ == '__main__':
    calibrate_camera()
