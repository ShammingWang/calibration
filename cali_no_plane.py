import cv2
import numpy as np
import glob
import json
import os

# 配置参数
chessboard_size = (6,8)  # 棋盘格规格：8 列 6 行 交点


# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 存储物理坐标和像素坐标
objpoints = []  # 世界坐标系下的物理坐标
imgpoints = []  # 图像坐标系下的像素坐标

# 加载标定图像和对应的物理坐标文件
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

    # 读取真实物理坐标
    objp = []
    with open(mocap_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 去掉空行
                continue
            x, y, z = map(float, line.split())
            objp.append([x, y, z])
    objp = np.array(objp, dtype=np.float32)
    
    # 检测棋盘格角点
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)  # 添加物理坐标
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)  # 添加亚像素坐标
        
        # 显示角点检测结果
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(2000)

cv2.destroyAllWindows()

# 确保有足够的数据点
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("没有足够的有效数据进行标定，请检查输入！")
    exit()


# 计算投影误差
def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """
    计算标定投影误差
    :param objpoints: 3D物理坐标
    :param imgpoints: 2D图像坐标
    :param rvecs: 旋转向量
    :param tvecs: 平移向量
    :param mtx: 相机内参矩阵
    :param dist: 畸变系数
    :return: 平均投影误差
    """
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        # 使用内外参将3D点投影到图像平面
        imgpoints_projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        # 计算欧几里得距离误差
        error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        total_error += error
        total_points += len(objpoints[i])

    mean_error = total_error / len(objpoints)  # 平均投影误差
    return mean_error

# # 设置初始内参矩阵
# image_size = gray.shape[::-1]  # 图像宽高 (width, height)
# fx = fy = image_size[0] * 1.2  # 假设焦距与图像宽度成比例
# cx = image_size[0] / 2          # 主点在图像的水平中心
# cy = image_size[1] / 2          # 主点在图像的垂直中心
# initial_camera_matrix = np.array([
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0,  0,  1]
# ], dtype=np.float32)

# 手动初始化内参矩阵
initial_camera_matrix = np.array([
    [1112.3583204903152, 0.0, 997.9270814766058],
    [0.0, 1113.7699122064623, 544.3827633351381],
    [0.0, 0.0, 1.0]
], dtype=np.float32)


# 手动初始化畸变参数
initial_dist_coeffs = np.array([
    -0.4271525642025255,
    0.22305966927928744,
    0.0017148576884110766,
    -0.0003402281805248362,
    -0.06510616799200525
], dtype=np.float32)

print("初始内参矩阵:\n", initial_camera_matrix)
print("手动设置的初始畸变参数:\n", initial_dist_coeffs)



# 执行相机标定
try:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], initial_camera_matrix, initial_dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC
    )

    print("内参矩阵:\n", mtx)
    print("畸变系数:\n", dist)

    # 计算投影误差
    reprojection_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print(f"平均投影误差: {reprojection_error:.6f} 像素")

    # 保存结果
    camera_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "extrinsics": []
    }

    for i in range(len(rvecs)):
        R, _ = cv2.Rodrigues(rvecs[i])
        extrinsic_matrix = np.hstack((R, tvecs[i]))
        camera_data["extrinsics"].append(extrinsic_matrix.tolist())
        print(f"图像 {i + 1} 的外参矩阵（3x4）：\n", extrinsic_matrix)

    output_file = "camera_calibration_test.json"
    with open(output_file, "w") as f:
        json.dump(camera_data, f, indent=4)
    print(f"标定结果已保存到文件：{output_file}")

except cv2.error as e:
    print("相机标定失败，请检查数据是否有效！")
    print("错误信息：", e)
