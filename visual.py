import cv2
import numpy as np
import json

# 从 JSON 文件中加载相机内参和畸变参数
def load_camera_calibration(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(data["dist_coeffs"][0], dtype=np.float32)
    return camera_matrix, dist_coeffs

# 初始化摄像头
def initialize_camera():
    cap = cv2.VideoCapture(2)  # 打开默认摄像头（设备索引 0）
    if not cap.isOpened():
        print("无法打开摄像头！")
        exit()

    # 设置摄像头分辨率为 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 验证分辨率设置是否成功
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width != 1920 or actual_height != 1080:
        print(f"警告: 无法将分辨率设置为 1920x1080，当前分辨率为 {actual_width}x{actual_height}")
    else:
        print("摄像头分辨率已成功设置为 1920x1080")
    return cap

# 实时显示矫正前后图像，支持缩放
def visualize_undistortion(cap, camera_matrix, dist_coeffs, scale=0.7):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧！")
            break

        # 获取图像尺寸
        h, w = frame.shape[:2]

        # 计算新的相机矩阵和矫正区域
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

        # 矫正图像
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # 显示原始图像和矫正图像
        combined = np.hstack((frame, undistorted_frame))

        # 调整显示窗口大小，保持比例
        display_height = int(combined.shape[0] * scale)
        display_width = int(combined.shape[1] * scale)
        resized_combined = cv2.resize(combined, (display_width, display_height))

        cv2.imshow('Original (Left) vs Undistorted (Right)', resized_combined)

        # 按 'ESC' 键退出，按 '+' 键放大，按 '-' 键缩小
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 键
            break
        elif key == ord('+'):  # 放大
            scale = min(2.0, scale + 0.1)  # 最大缩放 2 倍
        elif key == ord('-'):  # 缩小
            scale = max(0.3, scale - 0.1)  # 最小缩放 0.3 倍

# 主程序
def main():
    json_file = "camera_calibration.json"  # 替换为您的 JSON 文件路径
    camera_matrix, dist_coeffs = load_camera_calibration(json_file)
    print("相机内参矩阵:\n", camera_matrix)
    print("畸变系数:\n", dist_coeffs)

    # 初始化摄像头
    cap = initialize_camera()

    # 实时显示矫正效果
    try:
        visualize_undistortion(cap, camera_matrix, dist_coeffs)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    