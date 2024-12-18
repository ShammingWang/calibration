import cv2
import numpy as np
import json
import glob
import os

# 配置棋盘格参数
chessboard_size = (8, 6)  # 8列 6行交点

# 终止条件 (角点精细化)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 存储物理坐标和像素坐标
objpoints = []  # 世界坐标系下的物理坐标
imgpoints = []  # 图像坐标系下的像素坐标

# 1. 加载标定图像和物理坐标文件
image_files = glob.glob("calibration_images/*.jpg")
if not image_files:
    print("未找到标定图像，请检查 'calibration_images/' 文件夹！")
    exit()

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
            if not line:  # 跳过空行
                continue
            x, y, z = map(float, line.split())
            objp.append([x, y, z])
    objp = np.array(objp, dtype=np.float32)

    # 检测棋盘格角点
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

cv2.destroyAllWindows()

# 检查数据是否足够
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("没有足够的数据点进行验证！")
    exit()

# 2. 加载内参矩阵、畸变参数和外参矩阵
json_file = "camera_extrinsics.json"
with open(json_file, "r") as f:
    data = json.load(f)

camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
extrinsics = data["extrinsics"]

print("相机内参矩阵:\n", camera_matrix)
print("畸变参数:\n", dist_coeffs)

# 3. 计算投影误差
def calculate_projection_error(objpoints, imgpoints, extrinsic, camera_matrix, dist_coeffs):
    # 提取旋转矩阵 R 和平移向量 t，并转换数据类型
    R = np.array(extrinsic[:, :3], dtype=np.float64)  # 旋转矩阵 (3x3)
    t = np.array(extrinsic[:, 3], dtype=np.float64)   # 平移向量 (3x1)
    camera_matrix = camera_matrix.astype(np.float64)
    dist_coeffs = dist_coeffs.astype(np.float64)

    total_error = 0

    for i in range(len(objpoints)):
        # 确保 objpoints 是 float64 类型
        objpoints_i = objpoints[i].astype(np.float64)
        imgpoints_i = imgpoints[i].astype(np.float64)  # 转换为 np.float64

        # 使用内参、畸变参数和外参矩阵投影 3D 点到 2D 图像
        projected_points, _ = cv2.projectPoints(objpoints_i, R, t, camera_matrix, dist_coeffs)
        
        # 计算误差 (L2范数)
        error = cv2.norm(imgpoints_i, projected_points, cv2.NORM_L2) / len(projected_points)
        total_error += error

    return total_error / len(objpoints)


# 遍历每个外参矩阵，计算误差
errors = []
for idx, extrinsic in enumerate(extrinsics):
    extrinsic_matrix = np.array(extrinsic, dtype=np.float32)
    error = calculate_projection_error(objpoints, imgpoints, extrinsic_matrix, camera_matrix, dist_coeffs)
    errors.append(error)
    print(f"外参矩阵 {idx + 1} 的投影误差: {error:.6f} 像素")

# 输出最小误差的外参矩阵
best_idx = np.argmin(errors)
print(f"\n投影误差最小的是外参矩阵 {best_idx + 1}，误差为 {errors[best_idx]:.6f} 像素")
