
# Camera Calibration Tool

This project provides a comprehensive tool for camera calibration, including both intrinsic and extrinsic parameter calculations. The main script, `main_cali_all.py`, automates the calibration process and generates two output files: one for intrinsic parameters and another for extrinsic parameters.

## Features

- **Intrinsic Calibration**: Computes the camera matrix and distortion coefficients using chessboard images.
- **Extrinsic Calibration**: Computes the rotation and translation matrices for each image using corresponding 3D points.
- **Output Files**:
  - `intrinsic.json`: Contains the intrinsic camera parameters.
  - `extrinsics.json`: Contains the extrinsic parameters and reprojection errors.

## Input Requirements

The project requires a unified folder structure for input data. The folder should be organized as follows:

```
CalibrationInputFolder/
├── intrinsic_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── extrinsic_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── mocap_points/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### Folder Details

1. **`CalibrationInputFolder/intrinsic_images/`**:
   - Contains images in `.jpg` format.
   - These images are used for intrinsic calibration.
   - Ensure the images contain a visible chessboard pattern.

2. **`CalibrationInputFolder/extrinsic_images/`**:
   - Contains images in `.jpg` format.
   - These images are used for extrinsic calibration.
   - Ensure the images correspond to the 3D points provided in the `mocap_points` folder.

3. **`CalibrationInputFolder/mocap_points/`**:
   - Contains `.txt` files with the same names as the images in `extrinsic_images/`.
   - Each `.txt` file contains 3D coordinates of points in space, with one point per line.
   - Each line should have three floating-point numbers separated by spaces, representing the X, Y, and Z coordinates of a point.

### Example `mocap_points` File (`image1.txt`):
```
0.0 0.0 0.0
60.0 0.0 0.0
120.0 0.0 0.0
...
```

## How to Run

1. Ensure the input folder is structured as described above.
2. Place the input folder (e.g., `CalibrationInputFolder`) in the project directory.
3. Run the main script:
   ```bash
   python main_cali_all.py
   ```
4. The script will:
   - Compute intrinsic parameters using images from `intrinsic_images/`.
   - Compute extrinsic parameters using images from `extrinsic_images/` and corresponding 3D points from `mocap_points/`.
   - Save the results to `CalibrationInputFolder/intrinsic.json` and `CalibrationInputFolder/extrinsics.json`.

## Output Files

1. **`intrinsic.json`**:
   - Contains the camera matrix and distortion coefficients.
   - Example:
     ```json
     {
         "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
         "dist_coeffs": [k1, k2, p1, p2, k3]
     }
     ```

2. **`extrinsics.json`**:
   - Contains the extrinsic parameters (rotation and translation matrices) for each image.
   - Includes reprojection errors for quality assessment.
   - Example:
     ```json
     {
         "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
         "dist_coeffs": [k1, k2, p1, p2, k3],
         "extrinsics": [
             [[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3]],
             ...
         ],
         "reprojection_errors": [0.5, 0.6, ...],
         "best_extrinsic": [[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3]],
         "best_error": 0.5
     }
     ```

## Notes

- Ensure that the chessboard pattern in the images is clearly visible and consistent across all images.
- At least three images are required for both intrinsic and extrinsic calibration.
- The script will automatically create subfolders for intermediate results, such as undistorted images and visualized chessboard corners.

## Example Input Folder Structure

An example input folder structure is provided in `CalibrationInputFolder`. Use this as a reference for organizing your data.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Install the required dependencies using:
```bash
pip install opencv-python numpy matplotlib
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or issues, please contact the project maintainer.