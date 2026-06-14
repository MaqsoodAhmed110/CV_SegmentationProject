# Pixel-to-MM Measurement Report

## Methodology
The final measurement pipeline integrates all previously developed components to convert pixel coordinates into real-world millimetres. The workflow is as follows:

1. **Camera Calibration (Un-distortion):** Images are un-distorted using the intrinsic camera matrix to correct lens curvature.
2. **Reference Object Detection:** A checkerboard with known dimensions (20.0 mm squares) is placed on the same plane as the target object. OpenCV's `findChessboardCorners` extracts the Euclidean pixel distance between corners to dynamically calculate the `pixels_per_mm` scale for each image.
3. **Instance Segmentation:** The trained Detectron2 model generates a precise polygon mask of the tissue box.
4. **Dimension Extraction:** A minimum area bounding rectangle (`cv2.minAreaRect`) is fitted over the segmentation mask to calculate the maximum pixel width and height while accounting for object rotation.
5. **Metric Conversion:** The pixel dimensions are divided by the `pixels_per_mm` ratio to obtain the final real-world measurements.

## Measurement Output Examples

### Sample Result 1
![Measurement Output 1](1.png)

*Segmentation mask and measured dimensions overlaid on the test image.*

### Sample Result 2
![Measurement Output 2](2.png)

*Example demonstrating the pixel-to-millimetre conversion pipeline.*

## The Importance of Calibration
During testing, we observed that using raw, distorted images produces highly inconsistent measurements. Lens distortion (especially radial distortion near the edges) stretches pixels non-uniformly. Consequently, 100 pixels at the center of the image do not represent the same physical distance as 100 pixels at the edge. Intrinsic calibration mathematically normalizes this space, which is an absolute requirement for consistent metric derivation.

## Accuracy Validation

**Ground Truth Dimensions:** 101.6 mm (Width) × 50.8 mm (Height)

| Image Sample | Calculated Width (mm) | Calculated Height (mm) | Width Error (mm) | Height Error (mm) |
|:---|:---|:---|:---|:---|
| **Image 10** | 118.3 mm | 64.7 mm | 16.7 mm | 13.9 mm |
| **Image 1** | 89.1 mm | 53.3 mm | 12.5 mm | 2.5 mm |

### Final Error Statistics

| Metric | Result |
|:---|:---|
| **Mean Absolute Error (Width)** | 14.61 mm |
| **Mean Absolute Error (Height)** | 8.21 mm |
| **Mean Percentage Error (Width)** | 14.38% |
| **Mean Percentage Error (Height)** | 16.16% |

## Limitations and Future Improvements

While the system successfully demonstrates an end-to-end metric measurement pipeline, the ~14–16% Mean Percentage Error originates from two primary physical constraints:

1. **Perspective Distortion (Homography):** The camera was not perfectly parallel to the plane of the tissue box. Slight tilts cause the side of the box closer to the lens to appear larger. Future implementations should incorporate ArUco markers to compute a full homography matrix and warp the image into a top-down orthophoto prior to measurement.

2. **Calibration Sub-optimality:** The initial calibration phase relied on a limited number of valid viewpoints, meaning the un-distortion matrix could not perfectly correct extreme edge warping. Expanding the calibration dataset to 50+ diverse checkerboard angles would significantly reduce the measurement error.
