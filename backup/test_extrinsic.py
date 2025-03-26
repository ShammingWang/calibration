import cv2
import numpy as np
import glob
import json
import os

# Configuration Parameters
chessboard_size = (8, 6)  # Chessboard pattern size: 8 columns, 6 rows (inner corners)
images_path = "./calibration_images/*.jpg"
intrinsic_path = "./intrinsic.json"
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

# Load Intrinsic Parameters from JSON
camera_matrix, dist_coeffs = load_calibration_data(intrinsic_path)
print("Successfully loaded calibration parameters:")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Load Calibration Images and Corresponding 3D Object Points
image_files = glob.glob(images_path)
if not image_files:
    print("No calibration images found. Please check the 'calibration_images/' directory!")
    exit()

# Detect Chessboard Corners
for idx, image_file in enumerate(image_files):
    base_name = os.path.basename(image_file).split('.')[0]  # Remove extension
    mocap_file = os.path.join("mocap_points", f"{base_name}.txt")

    if not os.path.exists(mocap_file):
        print(f"Missing physical coordinates file: {mocap_file}. Skipping this image!")
        continue

    # Read 3D Object Points
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
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        # Uncomment the following lines if you wish to display the image with corners
        # cv2.imshow('Detected Corners', img)
        # cv2.waitKey(500)  # Display for 500 ms
        print(f"Image {idx + 1}: Chessboard corners detected and added.")
    else:
        print(f"Image {idx + 1}: Chessboard corners not detected. Skipping this image.")
        objpoints.pop()  # Remove the last appended objp as it's not matched
        continue

cv2.destroyAllWindows()

# Ensure Enough Data Points
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Insufficient valid data for calibration. Please check the inputs!")
    exit()

# Compute Extrinsic Parameters for Each Image
extrinsics = []
for i in range(len(objpoints)):
    ret, rvec, tvec = cv2.solvePnP(
        objpoints[i], imgpoints[i], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if ret:
        R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
        extrinsic_matrix = np.hstack((R, tvec))
        extrinsics.append(extrinsic_matrix.tolist())
        print(f"Image {i + 1} Extrinsic Matrix (3x4):\n", extrinsic_matrix)
    else:
        print(f"Image {i + 1}: solvePnP failed!")
        extrinsics.append(None)  # Mark failure

# Compute Reprojection Errors Using Corresponding Extrinsic Parameters
errors = []
for i, extrinsic in enumerate(extrinsics):
    if extrinsic is None:
        errors.append(float('inf'))  # Failure case
        continue

    # Extract Rotation Matrix and Translation Vector
    R = np.array(extrinsic)[:, :3]
    tvec = np.array(extrinsic)[:, 3]

    # Convert Rotation Matrix back to Rotation Vector for projectPoints
    rvec, _ = cv2.Rodrigues(R)

    # Project the 3D object points of the current image using its extrinsic parameters
    projected_points, _ = cv2.projectPoints(objpoints[i], rvec, tvec, camera_matrix, dist_coeffs)

    # Compute the reprojection error between the detected 2D points and the projected 3D points
    error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
    errors.append(error)
    print(f"Image {i + 1} Average Reprojection Error: {error:.6f} pixels")

# Identify the Extrinsic Matrix with the Minimum Reprojection Error
valid_errors = [error for error in errors if np.isfinite(error)]
if valid_errors:
    min_error_idx = np.argmin(valid_errors)
    print(f"\nExtrinsic Matrix with Minimum Error: Image {min_error_idx + 1}, Error = {valid_errors[min_error_idx]:.6f} pixels")
else:
    print("\nNo valid reprojection errors found.")
    min_error_idx = None

# Save Results to JSON
camera_data = {
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.tolist(),
    "extrinsics": extrinsics,
    "reprojection_errors": errors,
    "best_extrinsic": extrinsics[min_error_idx] if min_error_idx is not None else None,
    "best_error": errors[min_error_idx] if min_error_idx is not None else None
}

output_file = "camera_extrinsics_all_error.json"
with open(output_file, "w") as f:
    json.dump(camera_data, f, indent=4)
print(f"Extrinsic matrices and reprojection errors saved to file: {output_file}")
