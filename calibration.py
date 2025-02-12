import cv2
import numpy as np
import glob
import json

# 配置参数
chessboard_size = (4, 6)  # 棋盘格规格：4 行，6 列交点
square_size = 60  # 单格大小（单位：毫米）

# 初始化棋盘格物理坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # 按实际尺寸缩放

print(objp)


# 存储物理坐标和像素坐标
objpoints = []  # 世界坐标系下的物理坐标
imgpoints = []  # 图像坐标系下的像素坐标

# 加载标定图像
images = glob.glob("calibration_images/*.jpg")
if not images:
    print("未找到标定图像，请检查 'calibration_images/' 文件夹！")
    exit()

# 检测棋盘格角点
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)  # 添加物理坐标
        imgpoints.append(corners)  # 添加像素坐标
        
        # 显示角点检测结果
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# 执行相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印并保存内参矩阵
print("内参矩阵:\n", mtx)
print("畸变系数:\n", dist)

# 保存内参到文件
camera_data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeffs": dist.tolist()
}

# 保存外参矩阵
extrinsics = []  # 用于保存所有标定图像的外参
for i in range(len(rvecs)):
    R, _ = cv2.Rodrigues(rvecs[i])  # 将旋转向量转换为旋转矩阵
    extrinsic_matrix = np.hstack((R, tvecs[i]))  # 拼接为 3x4 外参矩阵
    extrinsics.append(extrinsic_matrix.tolist())  # 转为列表格式方便保存
    
    print(f"图像 {i + 1} 的外参矩阵（3x4）：\n", extrinsic_matrix)

camera_data["extrinsics"] = extrinsics  # 添加外参矩阵到字典

# 保存到 JSON 文件
output_file = "camera_calibration.json"
with open(output_file, "w") as f:
    json.dump(camera_data, f, indent=4)

print(f"标定结果已保存到文件：{output_file}")
