import numpy as np
import cv2 as cv
import json

def load_calibration_from_json(calibration_file):
    """
    从 JSON 文件中加载相机标定参数。

    参数:
        calibration_file (str): 相机标定 JSON 文件的路径。

    返回:
        tuple: 包含重投影误差、相机内参矩阵、畸变系数。
    """
    try:
        with open(calibration_file, "r") as f:
            data = json.load(f)
            ret = data["ret"]
            mtx = np.array(data["mtx"])
            dist = np.array(data["dist"])
        return ret, mtx, dist
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return None

def main():
    calibration_file = "camera_calibration.json"  # 校准参数的 JSON 文件
    ret, mtx, dist = load_calibration_from_json(calibration_file)

    if mtx is None or dist is None:
        print("Failed to load calibration data.")
        return

    # 初始化摄像头
    cap = cv.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # 设置摄像头分辨率为 1920x1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Press 'esc' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the frame.")
            break
        
        # 矫正畸变
        h, w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, newcameramtx)

        # 裁剪图像
        x, y, w, h = roi
        undistorted_frame = undistorted_frame[y:y+h, x:x+w]

        # 显示原图和校正图
        cv.imshow("Original Frame (1920x1080)", frame)
        cv.imshow("Undistorted Frame", undistorted_frame)

        # 按 'esc' 键退出
        if cv.waitKey(1) & 0xFF == 27:
            break

    # 释放资源
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
