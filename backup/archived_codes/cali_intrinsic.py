import numpy as np
import cv2 as cv
import glob
import os
import json


# Define the number of inner corners (rows and columns) in the chessboard
chessboard_rows = 6  # Number of inner corners in rows
chessboard_cols = 8  # Number of inner corners in columns
chessboard_length_mm = 60
images_path = "./calibration_intrinsic_images/*.jpg"

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(cols-1, rows-1, 0)
objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)
objp *= chessboard_length_mm

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Load all images with .jpg extension
images = glob.glob(images_path)

if len(images) <= 0:
    print("there is no image in the folder")
    exit(0)

gray = cv.cvtColor(cv.imread(images[0]), cv.COLOR_BGR2GRAY)

if not os.path.exists("ChessboardCorners"):
    os.mkdir("ChessboardCorners")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

    # If found, add object points and refine image points
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (chessboard_cols, chessboard_rows), corners2, ret)
        # cv.imshow('Chessboard Corners', img)
        # cv.waitKey(500)
        base_name = os.path.basename(fname)
        dst_fame = os.path.join("ChessboardCorners", base_name)
        cv.imwrite(dst_fame, img)

cv.destroyAllWindows()

# start to calibrate
# gray.shape ==> (h, w) , -1 means reverse
# opencv function needs (x,y) format
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def save_calibration_to_json(ret, mtx, dist, rvecs, tvecs, output_file="intrinsic.json"):
    """
    将相机标定结果保存到 JSON 文件。

    参数:
        ret (float): 重投影误差。
        mtx (numpy.ndarray): 相机内参矩阵。
        dist (numpy.ndarray): 畸变系数。
        rvecs (list): 旋转向量的列表。
        tvecs (list): 平移向量的列表。
        output_file (str): 保存 JSON 文件的路径，默认是 "intrinsic.json"。

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
    
save_calibration_to_json(ret, mtx, dist, rvecs, tvecs, output_file="intrinsic.json")

if not os.path.exists("undistorted_images"):
    os.mkdir("undistorted_images")

for fname in images:
    img = cv.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    base_name = os.path.basename(fname)
    dst_fame = os.path.join("undistorted_images", base_name)
    cv.imwrite(dst_fame, dst)
