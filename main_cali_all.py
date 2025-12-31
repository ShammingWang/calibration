import numpy as np
import glob
import cv2
import os
import json
import matplotlib.pyplot as plt


# Define the number of inner corners (rows and columns) in the chessboard
chessboard_rows = 6  # Number of inner corners in rows
chessboard_cols = 8  # Number of inner corners in columns
chessboard_length_mm = 60
input_folder="./data/TopRight/"
intrinsic_images_path = os.path.join(input_folder, "extrinsic_images/*.jpg")
intrinsic_path = os.path.join(input_folder, "intrinsic.json")
extrinsic_images_path = os.path.join(input_folder, "extrinsic_images/*.jpg")

def calibrate_intrinsic(show_chessboard_corners=True):
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(cols-1, rows-1, 0)
    objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)
    objp *= chessboard_length_mm

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Load all images with .jpg extension
    images = glob.glob(intrinsic_images_path)

    if len(images) <= 0:
        print("there is no jpg image in the folder")
        exit(0)

    gray = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY)

    if show_chessboard_corners and not os.path.exists(os.path.join(input_folder, "ChessboardCorners")):
        os.mkdir(os.path.join(input_folder, "ChessboardCorners"))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

        # If found, add object points and refine image points
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if show_chessboard_corners:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (chessboard_cols, chessboard_rows), corners2, ret)
                # cv2.imshow('Chessboard Corners', img)
                # cv2.waitKey(500)
                base_name = os.path.basename(fname)
                dst_fame = os.path.join(input_folder, "ChessboardCorners", base_name)
                cv2.imwrite(dst_fame, img)

    # cv2.destroyAllWindows()

    # start to calibrate
    # gray.shape ==> (h, w) , -1 means reverse
    # opencv2 function needs (x,y) format
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def save_calibration_to_json(ret, mtx, dist, rvecs, tvecs, output_file):
        """
        将相机标定结果保存到 JSON 文件。

        参数:
            ret (float): 重投影误差。
            mtx (numpy.ndarray): 相机内参矩阵。
            dist (numpy.ndarray): 畸变系数。
            rvecs (list): 旋转向量的列表。
            tvecs (list): 平移向量的列表。
            output_file (str): 保存 JSON 文件的路径。

        返回:
            bool: 保存成功返回 True，否则返回 False。
        """
        try:
            # 将标定结果转换为 JSON 格式
            calibration_data = {
                # "ret": ret,  # 重投影误差
                "camera_matrix": mtx.tolist(),  # 相机内参矩阵
                "dist_coeffs": dist.tolist(),  # 畸变系数
                # "rvecs": [rvec.flatten().tolist() for rvec in rvecs],  # 旋转向量
                # "tvecs": [tvec.flatten().tolist() for tvec in tvecs],  # 平移向量
            }

            # 保存结果到 JSON 文件
            with open(output_file, "w") as f:
                json.dump(calibration_data, f, indent=4)

            print(f"Camera calibration data has been saved to {output_file}")
            return True

        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    save_calibration_to_json(ret, mtx, dist, rvecs, tvecs, output_file=os.path.join(input_folder, "intrinsic.json"))

    if not os.path.exists(os.path.join(input_folder, "undistorted_images")):
        os.mkdir(os.path.join(input_folder, "undistorted_images"))

    for fname in images:
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        base_name = os.path.basename(fname)
        dst_fame = os.path.join(input_folder, "undistorted_images", base_name)
        cv2.imwrite(dst_fame, dst)

# calculate the extrinsic parameters
def calibrate_extrinsics(images_path=extrinsic_images_path, intrinsic_path=intrinsic_path, 
                         chessboard_size=(chessboard_cols, chessboard_rows)):

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def load_calibration_data(json_file):
        with open(json_file, "r") as f:
            calibration_data = json.load(f)
        mtx = np.array(calibration_data["camera_matrix"], dtype=np.float32)
        dist = np.array(calibration_data["dist_coeffs"], dtype=np.float32)
        return mtx, dist

    def calculate_reprojection_error(objpoints, imgpoints, rvec, tvec, camera_matrix, dist_coeffs):
        projected_points, _ = cv2.projectPoints(objpoints, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints, projected_points, cv2.NORM_L2) / len(projected_points)
        return error

    # Load intrinsic parameters
    camera_matrix, dist_coeffs = load_calibration_data(intrinsic_path)
    print("Successfully loaded calibration parameters:")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # 直接使用内参计算好的参数 后续计算外参保持不变
    calibrated_camera_matrix = camera_matrix
    calibrated_dist_coeffs = dist_coeffs


    # Load calibration images
    image_files = glob.glob(images_path)
    if not image_files:
        print("No calibration images found. Please check the 'calibration_images/' directory!")
        return

    objpoints = []  # 3D points in the world coordinate system
    imgpoints = []  # 2D points in the image plane

    for idx, image_file in enumerate(image_files):
        base_name = os.path.basename(image_file).split('.')[0]
        
        mocap_file = os.path.join(input_folder, f"mocap_points/sorted/{base_name}.txt")

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

        objpoints.append(objp)

        # Load image and detect chessboard corners
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to load image: {image_file}. Skipping!")
            objpoints.pop()
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"Image {idx + 1}: Chessboard corners detected and added.")
        else:
            print(f"Image {idx + 1}: Chessboard corners not detected. Skipping this image.")
            objpoints.pop()
            continue

    if len(objpoints) < 3 or len(imgpoints) < 3:
        print("Insufficient valid data for calibration. At least 3 images are required!")
        return

    # # Perform camera calibration
    # ret, calibrated_camera_matrix, calibrated_dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    #     objpoints, imgpoints, gray.shape[::-1], camera_matrix, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS
    # )

    # print("\nCalibration Results:")
    # print("Calibrated Camera Matrix:\n", calibrated_camera_matrix)
    # print("\nCalibrated Distortion Coefficients:\n", calibrated_dist_coeffs.ravel())

    # Compute extrinsic parameters and reprojection errors
    extrinsics = []
    errors = []
    for i in range(len(objpoints)):
        success, rvec, tvec = cv2.solvePnP(
            objpoints[i], imgpoints[i], calibrated_camera_matrix, calibrated_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)
            extrinsic_matrix = np.hstack((R, tvec))
            extrinsics.append(extrinsic_matrix.tolist())

            reprojection_error = calculate_reprojection_error(objpoints[i], imgpoints[i], rvec, tvec, calibrated_camera_matrix, calibrated_dist_coeffs)
            errors.append(reprojection_error)
            print(f"\nImage {i + 1} Extrinsic Matrix (3x4):\n", extrinsic_matrix)
            print(f"Image {i + 1} Reprojection Error: {reprojection_error:.6f} pixels")
        else:
            print(f"\nImage {i + 1}: solvePnP failed!")
            extrinsics.append(None)
            errors.append(float('inf'))

    valid_errors = [error for error in errors if np.isfinite(error)]
    if valid_errors:
        min_error_idx = np.argmin(valid_errors)
        print(f"\nExtrinsic Matrix with Minimum Error: Image {min_error_idx + 1}, Error = {valid_errors[min_error_idx]:.6f} pixels")
    else:
        print("\nNo valid reprojection errors found.")
        min_error_idx = None

    camera_data = {
        "camera_matrix": calibrated_camera_matrix.tolist(),
        "dist_coeffs": calibrated_dist_coeffs.ravel().tolist(),
        "extrinsics": extrinsics,
        "reprojection_errors": errors,
        "best_extrinsic": extrinsics[min_error_idx] if min_error_idx is not None else None,
        "best_error": errors[min_error_idx] if min_error_idx is not None else None
    }
    
    output_file = os.path.join(input_folder, "extrinsics.json")
    with open(output_file, "w") as f:
        json.dump(camera_data, f, indent=4)
    print(f"\nExtrinsic matrices and reprojection errors saved to file: {output_file}")


# Example usage:
# calibrate_intrinsic()
# calibrate_extrinsics()
if __name__ == "__main__":
    calibrate_intrinsic(show_chessboard_corners=True)
    calibrate_extrinsics()
