import cv2 as cv
import numpy as np
import json
import os

def undistort_images(input_folder, json_file, output_folder):
    """
    读取指定文件夹中的所有 JPG 图片，利用 JSON 文件中的相机内参进行畸变矫正，
    并将矫正后的图片保存到指定的文件夹中。

    参数:
        input_folder (str): 待读取的 JPG 图片所在的文件夹路径。
        json_file (str): 包含相机内参的 JSON 文件路径。
        output_folder (str): 保存矫正后图片的文件夹路径。如果没有目录，自动创建。
    """
    try:
        # 检查输出文件夹是否存在，如果不存在则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 读取相机内参和畸变系数
        with open(json_file, 'r') as f:
            calibration_data = json.load(f)
            mtx = np.array(calibration_data["camera_matrix"])  # Convert to numpy array
            dist = np.array(calibration_data["dist_coeffs"])   # Convert to numpy array

        # 获取输入文件夹中的所有 jpg 图片文件
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
        
        # 逐一读取并矫正每张图片
        for fname in image_files:
            img_path = os.path.join(input_folder, fname)
            img = cv.imread(img_path)
            h, w = img.shape[:2]
            
            # 获取优化后的相机矩阵
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            
            # 对图像进行畸变矫正
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            
            # 裁剪图像
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            
            # 保存矫正后的图像
            output_path = os.path.join(output_folder, fname)
            cv.imwrite(output_path, dst)
        
        print(f"All images have been undistorted and saved to {output_folder}.")
    
    except Exception as e:
        print(f"An error occurred: {e}")


undistort_images("calibration_images", "intrinsic.json", "undistorted_images")
