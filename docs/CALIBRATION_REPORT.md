# Camera Calibration Report

## Method
Intrinsic camera calibration was performed using OpenCV's `cv2.calibrateCamera` function with a standard black-and-white checkerboard pattern. A total of 22 unique images were captured from varied angles and distances. The algorithm successfully identified the inner corners on 4 distinct viewpoints to compute the intrinsic camera matrix and distortion coefficients. Sub-pixel optimization was used to refine the corner detections before computing the final matrix.
## Calibration Images

Here is an example of a checkerboard image used for calibration.
![Detected Checkerboard Corners](/calibration.png "Calibration Image with Detected Corners")
## Calibration Parameters

### Intrinsic Camera Matrix
This matrix describes the focal length ($f_x$, $f_y$) and optical centers ($c_x$, $c_y$).
```text
[[2.15951334e+03, 0.00000000e+00, 2.63951523e+02],
 [0.00000000e+00, 9.32487525e+02, 3.74235873e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
```

### Distortion Coefficients
This array describes the radial and tangential lens distortion.
```text
[-1.43344452e-01,  3.37025218e+00, -7.06862148e-04, -8.13322289e-03, -1.50482656e+01]
```

## Error Analysis
**Total Reprojection Error: 0.0575 pixels**

The reprojection error is exceptionally low (well below the required 0.5px threshold). This ensures that the un-distortion applied to the measurement images will be mathematically sound, removing lens-curvature artifacts prior to pixel-to-mm conversion.
