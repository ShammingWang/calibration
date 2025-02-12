import cv2
import datetime
import os

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率为 1920x1080 (1080p)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

# 验证设置是否成功
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"摄像头分辨率设置为: {int(width)}x{int(height)}")

print("按 'Q' 键拍摄图片并保存，按 'Esc' 键退出程序。")

if not os.path.exists("calibration_images"):
    os.makedirs("calibration_images")

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面！")
        break

    # 显示画面
    cv2.imshow("Live Camera", frame)

    # 等待键盘事件
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):  # 按 'Q' 键
        # 获取当前时间
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"calibration_images/{current_time}.jpg"

        # 保存图片
        cv2.imwrite(file_name, frame)
        print(f"图片已保存为：{file_name}")
    elif key == 27:  # 按 'Esc' 键退出
        print("退出程序。")
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
