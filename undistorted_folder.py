import cv2 as cv
import numpy as np
import os
import glob
import json

# 1. 读取 JSON 文件中的标定结果
def load_calibration_data(json_file):
    """
    从 JSON 文件中加载相机标定参数。

    参数:
        json_file (str): JSON 文件路径。

    返回:
        dict: 包含相机内参矩阵、畸变系数和其他参数的数据字典。
    """
    with open(json_file, "r") as f:
        calibration_data = json.load(f)
    
    mtx = np.array(calibration_data["mtx"], dtype=np.float32)  # 相机内参矩阵
    dist = np.array(calibration_data["dist"], dtype=np.float32)  # 畸变系数
    return mtx, dist

# 2. 矫正图像并保存
def undistort_images(input_folder, output_folder, mtx, dist):
    """
    使用标定参数矫正图像，并将结果保存到指定文件夹。

    参数:
        input_folder (str): 包含待矫正图像的文件夹路径。
        output_folder (str): 保存矫正后图像的文件夹路径。
        mtx (numpy.ndarray): 相机内参矩阵。
        dist (numpy.ndarray): 畸变系数。
    """
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取所有 JPG 图片
    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    if not images:
        print(f"在 {input_folder} 中未找到任何 JPG 图片！")
        return
    
    print(f"开始矫正 {len(images)} 张图片...")
    for fname in images:
        img = cv.imread(fname)
        h, w = img.shape[:2]

        # 计算新的相机矩阵
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # 矫正图像
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        
        # 裁剪图像
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # 保存矫正后的图像
        base_name = os.path.basename(fname)
        output_path = os.path.join(output_folder, base_name)
        cv.imwrite(output_path, dst)
        print(f"已保存矫正后的图像: {output_path}")
    
    print("所有图像已完成矫正！")

# 主程序
if __name__ == "__main__":
    # 1. 加载标定参数
    json_file = "instic_and_distort.json"  # 标定结果 JSON 文件
    input_folder = "calibration_images"  # 输入图片文件夹
    output_folder = "undistorted_images"  # 输出矫正后图片文件夹

    # 加载相机标定数据
    mtx, dist = load_calibration_data(json_file)
    print("成功加载相机标定数据！")
    print("相机内参矩阵:\n", mtx)
    print("畸变系数:\n", dist)

    # 2. 矫正图像并保存
    undistort_images(input_folder, output_folder, mtx, dist)
