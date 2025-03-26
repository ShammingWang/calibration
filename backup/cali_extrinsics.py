import cv2
import numpy as np
import glob
import json
import os
import matplotlib.pyplot as plt

# Configuration Parameters
images_path = "./calibration_images/*.jpg"  # Path to calibration images
intrinsic_path = "./intrinsic.json"         # Path to intrinsic parameters JSON
chessboard_size = (8, 6)                    # Chessboard pattern size: 8 columns, 6 rows (inner corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Storage for 3D and 2D points
objpoints = []  # 3D points in the world coordinate system
imgpoints = []  # 2D points in the image plane

def load_calibration_data(json_file):
    """
    Loads camera intrinsic parameters from a JSON file.
    
    :param json_file: Path to the JSON file containing intrinsic parameters.
    :return: Tuple of camera matrix and distortion coefficients.
    """
    with open(json_file, "r") as f:
        calibration_data = json.load(f)
    mtx = np.array(calibration_data["camera_matrix"], dtype=np.float32)
    dist = np.array(calibration_data["dist_coeffs"], dtype=np.float32)
    return mtx, dist

def calculate_reprojection_error(objpoints, imgpoints, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Calculates the average reprojection error for given extrinsic parameters.
    
    :param objpoints: Nx3 array of 3D object points.
    :param imgpoints: Nx2 array of 2D image points.
    :param rvec: Rotation vector.
    :param tvec: Translation vector.
    :param camera_matrix: Camera matrix.
    :param dist_coeffs: Distortion coefficients.
    :return: Average reprojection error in pixels.
    """
    projected_points, _ = cv2.projectPoints(objpoints, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints, projected_points, cv2.NORM_L2) / len(projected_points)
    return error


def visualize_corners(image, corners, chessboard_size, success, window_name='Chessboard Corners'):
    """
    Visualizes detected chessboard corners on the image.
    
    :param image: Image on which to draw corners.
    :param corners: Detected corner points.
    :param chessboard_size: Tuple indicating number of inner corners per chessboard row and column.
    :param success: Boolean indicating if corners were found.
    :param window_name: Name of the display window.
    """
    if success:
        cv2.drawChessboardCorners(image, chessboard_size, corners, success)
    cv2.imshow(window_name, image)
    cv2.waitKey(500)  # Display for 500 ms

def plot_points(points, sorted_points):
    """
    Plots the original and sorted points in 3D space with annotations.
    
    :param points: Original Nx3 array of 3D points.
    :param sorted_points: Sorted Nx3 array of 3D points.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Original Points')

    # Plot sorted points
    ax.scatter(sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2], 
               c='g', marker='^', label='Sorted Points')

    # Annotate sorted points with their order
    for i, point in enumerate(sorted_points):
        ax.text(point[0], point[1], point[2], f'{i + 1}', color='red', fontsize=8)

    # Label axes
    ax.set_xlabel('X (Horizontal)')
    ax.set_ylabel('Y (Vertical)')
    ax.set_zlabel('Z (Ground Parallel)')
    ax.set_title('3D Points Sorted in Row-Major Order')
    ax.legend()

    # Adjust the viewing angle for better visualization
    ax.view_init(elev=20, azim=30)  # Adjust elevation and azimuth as needed

    plt.show()

def main(json_file_path):
    """
    Main function to perform camera calibration.
    
    :param json_file_path: Path to the JSON file containing intrinsic parameters.
    """
    # Step 1: Load Camera Intrinsic Parameters
    camera_matrix, dist_coeffs = load_calibration_data(json_file_path)
    print("Successfully loaded calibration parameters:")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # Step 2: Load Calibration Images
    image_files = glob.glob(images_path)
    if not image_files:
        print("No calibration images found. Please check the 'calibration_images/' directory!")
        exit()

    # Step 3: Detect Chessboard Corners and Collect Points
    for idx, image_file in enumerate(image_files):
        base_name = os.path.basename(image_file).split('.')[0]  # Remove extension
        mocap_file = os.path.join("mocap_points", f"{base_name}.txt")

        if not os.path.exists(mocap_file):
            print(f"Missing physical coordinates file: {mocap_file}. Skipping this image!")
            continue

        # Read 3D object points from mocap file
        objp = []
        with open(mocap_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                x, y, z = map(float, line.split())
                objp.append([x, y, z])
        objp = np.array(objp, dtype=np.float32)

        expected_points = chessboard_size[0] * chessboard_size[1]
        if objp.shape[0] != expected_points:
            print(f"Incorrect number of 3D points in {mocap_file}. Expected {expected_points}, got {objp.shape[0]}. Skipping!")
            continue

        # Sort 3D points into row-major order
        objpoints.append(objp)

        # Load image and detect chessboard corners on the original image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to load image: {image_file}. Skipping!")
            objpoints.pop()  # Remove last objp as it's not matched
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Refine corner locations for better accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Visualize detected corners
            visualize_corners(img.copy(), corners2, chessboard_size, ret, window_name='Detected Corners')
            print(f"Image {idx + 1}: Chessboard corners detected and added.")
        else:
            print(f"Image {idx + 1}: Chessboard corners not detected. Skipping this image.")
            objpoints.pop()  # Remove the last appended objp as it's not matched
            continue

    cv2.destroyAllWindows()

    # Step 4: Ensure Enough Data Points
    if len(objpoints) < 3 or len(imgpoints) < 3:
        print("Insufficient valid data for calibration. At least 3 images are required!")
        exit()

    # Step 5: Perform Camera Calibration
    ret, calibrated_camera_matrix, calibrated_dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], camera_matrix, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    print("\nCalibration Results:")
    print("Calibrated Camera Matrix:\n", calibrated_camera_matrix)
    print("\nCalibrated Distortion Coefficients:\n", calibrated_dist_coeffs.ravel())

    # Step 6: Compute Extrinsic Parameters and Reprojection Errors for Each Image
    extrinsics = []
    errors = []  # Store reprojection errors for each extrinsic
    for i in range(len(objpoints)):
        success, rvec, tvec = cv2.solvePnP(
            objpoints[i], imgpoints[i], calibrated_camera_matrix, calibrated_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
            extrinsic_matrix = np.hstack((R, tvec))
            extrinsics.append(extrinsic_matrix.tolist())

            # Calculate reprojection error
            reprojection_error = calculate_reprojection_error(objpoints[i], imgpoints[i], rvec, tvec, calibrated_camera_matrix, calibrated_dist_coeffs)
            errors.append(reprojection_error)
            print(f"\nImage {i + 1} Extrinsic Matrix (3x4):\n", extrinsic_matrix)
            print(f"Image {i + 1} Reprojection Error: {reprojection_error:.6f} pixels")
        else:
            print(f"\nImage {i + 1}: solvePnP failed!")
            extrinsics.append(None)  # Mark failure
            errors.append(float('inf'))

    # Step 7: Identify Extrinsic with Minimum Reprojection Error
    valid_errors = [error for error in errors if np.isfinite(error)]
    if valid_errors:
        min_error_idx = np.argmin(valid_errors)
        print(f"\nExtrinsic Matrix with Minimum Error: Image {min_error_idx + 1}, Error = {valid_errors[min_error_idx]:.6f} pixels")
    else:
        print("\nNo valid reprojection errors found.")
        min_error_idx = None

    # Step 8: Save Results to JSON
    camera_data = {
        "camera_matrix": calibrated_camera_matrix.tolist(),
        "dist_coeffs": calibrated_dist_coeffs.ravel().tolist(),
        "extrinsics": extrinsics,
        "reprojection_errors": errors,
        "best_extrinsic": extrinsics[min_error_idx] if min_error_idx is not None else None,
        "best_error": errors[min_error_idx] if min_error_idx is not None else None
    }

    output_file = "camera_extrinsics_single_error.json"
    with open(output_file, "w") as f:
        json.dump(camera_data, f, indent=4)
    print(f"\nExtrinsic matrices and reprojection errors saved to file: {output_file}")

if __name__ == "__main__":
    # Path to the JSON file containing intrinsic parameters
    if not os.path.exists(intrinsic_path):
        print(f"Intrinsic parameters file '{intrinsic_path}' not found. Please provide the correct path.")
        exit()
    main(intrinsic_path)
