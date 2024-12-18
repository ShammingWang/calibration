import cv2
import numpy as np
import glob
import json
import os

# 配置参数
chessboard_size = (9 - 1, 7 - 1)  # 棋盘格规格：8 列 6 行 交点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 存储物理坐标和像素坐标
objpoints = []  # 世界坐标系下的物理坐标
imgpoints = []  # 图像坐标系下的像素坐标

# 读取标定结果 JSON 文件
def load_calibration_data(json_file):
    with open(json_file, "r") as f:
        calibration_data = json.load(f)
    mtx = np.array(calibration_data["camera_matrix"], dtype=np.float32)
    dist = np.array(calibration_data["dist_coeffs"], dtype=np.float32)
    return mtx, dist

# 计算投影误差
def calculate_reprojection_error(objpoints, imgpoints, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(objpoints, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints, projected_points, cv2.NORM_L2) / len(projected_points)
    return error

# JSON 文件中的标定参数
json_file = "intrinsic.json"
camera_matrix, dist_coeffs = load_calibration_data(json_file)
print("成功加载标定参数：")
print("相机内参矩阵:\n", camera_matrix)
print("畸变参数:\n", dist_coeffs)

# 加载标定图像和对应的物理坐标文件
image_files = glob.glob("calibration_images/*.jpg")
if not image_files:
    print("未找到标定图像，请检查 'calibration_images/' 文件夹！")
    exit()

# 检测棋盘格角点
for image_file in image_files:
    base_name = os.path.basename(image_file).replace('.jpg', '')
    mocap_file = os.path.join("mocap_points", f"{base_name}.txt")

    if not os.path.exists(mocap_file):
        print(f"未找到物理坐标文件：{mocap_file}，跳过该图像！")
        continue

    # 读取物理坐标
    objp = []
    with open(mocap_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y, z = map(float, line.split())
            objp.append([x, y, z])
    objp = np.array(objp, dtype=np.float32)

    # 加载图像并去畸变
    img = cv2.imread(image_file)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)

    # 裁剪图像
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    # 转换为灰度图并检测棋盘格角点
    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        # 加入亚像素优化的话 对于这种标定板特别小的情况 会出现严重的偏移 
        # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        
        # 可视化角点检测结果
        cv2.drawChessboardCorners(undistorted_img, chessboard_size, corners, ret)
        # cv2.drawChessboardCorners(undistorted_img, chessboard_size, corners2, ret)
        # cv2.namedWindow('Undistorted Chessboard Corners', cv2.WINDOW_FREERATIO)
        # cv2.imshow('Undistorted Chessboard Corners', undistorted_img)
        # cv2.imwrite("findChessboardCornenr2.jpg", undistorted_img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

# 确保有足够的数据点
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("没有足够的有效数据进行外参计算，请检查输入！")
    exit()


# 使用 solvePnP 计算所有外参矩阵
extrinsics = []
for i in range(len(objpoints)):
    ret, rvec, tvec = cv2.solvePnP(
        objpoints[i], imgpoints[i], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if ret:
        R, _ = cv2.Rodrigues(rvec)  # 旋转向量转换为旋转矩阵
        extrinsic_matrix = np.hstack((R, tvec))
        extrinsics.append(extrinsic_matrix.tolist())
        print(f"图像 {i + 1} 的外参矩阵（3x4）：\n", extrinsic_matrix)
    else:
        print(f"图像 {i + 1} 的 solvePnP 计算失败！")
        extrinsics.append(None)  # 标记失败


# 计算每个外参矩阵对应的平均投影误差
errors = []
for i, extrinsic in enumerate(extrinsics):
    if extrinsic is None:
        errors.append(float('inf'))  # 失败的情况
        continue

    # 提取旋转矩阵和平移向量
    R = np.array(extrinsic)[:, :3]
    tvec = np.array(extrinsic)[:, 3]

    # 计算所有点的平均投影误差
    total_error = 0
    total_points = 0
    for j in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(objpoints[j], cv2.Rodrigues(R)[0], tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[j], projected_points, cv2.NORM_L2) / len(projected_points)
        total_error += error
        total_points += 1

    avg_error = total_error / total_points
    errors.append(avg_error)
    print(f"图像 {i + 1} 的平均投影误差: {avg_error:.6f} 像素")

# 找出误差最小的外参矩阵
min_error_idx = np.argmin(errors)
print(f"\n误差最小的外参矩阵为图像 {min_error_idx + 1}，误差为 {errors[min_error_idx]:.6f} 像素")

# 保存结果
camera_data = {
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.tolist(),
    "extrinsics": extrinsics,
    "reprojection_errors": errors,
    "best_extrinsic": extrinsics[min_error_idx],
    "best_error": errors[min_error_idx]
}

output_file = "camera_extrinsics_all_error.json"
with open(output_file, "w") as f:
    json.dump(camera_data, f, indent=4)
print(f"外参矩阵及平均投影误差结果已保存到文件：{output_file}")
