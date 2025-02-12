import numpy as np
import pandas as pd
import csv
import json
import cv2

# -------------------------------------------------------
# Example/demo code to show how one might:
#  1) Read a truncated CSV with 3D keypoints from OptiTrack
#  2) Map them to Human 3.6M joint indices
#  3) Project them onto 2D video frames using camera intrinsics/extrinsics
#  4) Overlay (draw) the projected points on top of the video
#
# NOTE: This is a simplified example to illustrate the pipeline;
#       adapt to your real data and naming conventions.
# -------------------------------------------------------

# H36M joint mapping from your description
# (showing final list of relevant indices for convenience)
H36M_NAMES = [""] * 32
H36M_NAMES[0]  = "Hip"
H36M_NAMES[1]  = "RHip"
H36M_NAMES[2]  = "RKnee"
H36M_NAMES[3]  = "RFoot"
H36M_NAMES[4]  = "LHip"
H36M_NAMES[5]  = "LKnee"
H36M_NAMES[6]  = "LFoot"
H36M_NAMES[7]  = "Spine"
H36M_NAMES[8]  = "Thorax"
H36M_NAMES[9]  = "Neck/Nose"
H36M_NAMES[10] = "Head"
H36M_NAMES[11] = "LShoulder"
H36M_NAMES[12] = "LElbow"
H36M_NAMES[13] = "LWrist"
H36M_NAMES[14] = "RShoulder"
H36M_NAMES[15] = "RElbow"
H36M_NAMES[16] = "RWrist"
# ... indices 17..31 omitted for brevity

# Suppose you want just these 17 indices:
TARGET_JOINT_INDICES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# This dict maps each H3.6M index to one or more marker names in your CSV
TARGET_JOINTS_ORDERED = {
    0:  ['Skeleton 002:Hip'], 
    1:  ['Skeleton 002:WaistRBack','Skeleton 002:WaistLBack'],
    2:  ['Skeleton 002:RKneeOut'],
    3:  ['Skeleton 002:RFoot'],
    4:  ['Skeleton 002:WaistLBack','Skeleton 002:WaistLFront'],
    5:  ['Skeleton 002:LKneeOut'],
    6:  ['Skeleton 002:LFoot'],
    7:  ['Skeleton 002:BackLeft','Skeleton 002:BackRight'],
    8:  ['Skeleton 002:BackTop'],
    9:  ['Skeleton 002:Neck'],
    10: ['Skeleton 002:Head'],
    11: ['Skeleton 002:LShoulder'],
    12: ['Skeleton 002:LElbowOut'],
    13: ['Skeleton 002:LWristIn','Skeleton 002:LWristOut'],
    14: ['Skeleton 002:RShoulder'],
    15: ['Skeleton 002:RElbowOut'],
    16: ['Skeleton 002:RWristIn','Skeleton 002:RWristOut']
}


def load_camera_parameters_json(intrinsic_path, extrinsic_path):
    """
    Loads camera intrinsic and extrinsic from JSON files.
    Returns (camera_matrix, dist_coeffs, best_extrinsic) as np arrays.
    """
    with open(intrinsic_path, "r") as f:
        intrinsic_data = json.load(f)
        camera_matrix = np.array(intrinsic_data["camera_matrix"])
        dist_coeffs   = np.array(intrinsic_data["dist_coeffs"])
    
    with open(extrinsic_path, "r") as f:
        extrinsic_data = json.load(f)
        # Let's pick out the "best_extrinsic" for demonstration
        best_extrinsic = np.array(extrinsic_data["best_extrinsic"])
    
    return camera_matrix, dist_coeffs, best_extrinsic


def project_points_3d_to_2d(keypoints_3d, camera_matrix, dist_coeffs, extrinsic_3x4):
    """
    Projects 3D points into 2D pixel coordinates (OpenCV style).
    - keypoints_3d: Nx3 array of 3D points in the camera's coordinate system
    - camera_matrix: 3x3
    - dist_coeffs: 1x5 or similar
    - extrinsic_3x4: The rotation+translation matrix from world to camera (or a suitable transformation)
    Returns Nx2 array of 2D keypoints in pixel space.
    """
    # Convert extrinsic from shape (3,4) to full 4x4 for matrix multiply convenience
    extrinsic_4x4 = np.eye(4)
    extrinsic_4x4[:3, :4] = extrinsic_3x4

    # Convert keypoints_3d to homogeneous coords: shape (N,4)
    N = keypoints_3d.shape[0]
    ones = np.ones((N,1))
    homogeneous_3d = np.hstack((keypoints_3d, ones))  # [x, y, z, 1]
    
    # Transform points from world to camera
    points_camera = (extrinsic_4x4 @ homogeneous_3d.T).T  # shape (N,4)
    # Discard last column, keep (x, y, z) in camera coords
    points_camera = points_camera[:, :3]

    # Now we have (Xc, Yc, Zc), use OpenCV's projectPoints
    # to handle lens distortion + intrinsics
    rvec = np.zeros((3,1))  # assuming extrinsic_4x4 includes rotation
    tvec = np.zeros((3,1))  # and translation, so no separate here
    
    # OpenCV expects shape=(N,1,3)
    points_camera_cv = points_camera.reshape(-1,1,3).astype(np.float32)
    
    projected_pts, _ = cv2.projectPoints(
        points_camera_cv, rvec, tvec, camera_matrix, dist_coeffs
    )
    # projected_pts will be Nx1x2
    projected_pts = projected_pts.squeeze()  # becomes Nx2
    return projected_pts


def parse_optitrack_csv(csv_path):
    """
    Reads (and lightly parses) an OptiTrack CSV:
      - Skips metadata rows if needed
      - Locates columns for each bone/marker
      - Returns a (17, 3) array of 3D coords or (num_frames, 17, 3).
    
    For simplicity, the example returns a single frame of 17 joints.
    If your data has multiple frames, you can read them all and stack them.
    """
    # Adjust skiprows to skip the lines that contain "NaN, Name, ID, Rotation" etc.
    # If row 3 is your 'Frame, Time (Seconds), X, Y, Z' heading, you might do skiprows=3
    df = pd.read_csv(csv_path, skiprows=2)

    # df now presumably has columns like:
    # ["Frame", "Time (Seconds)", "Skeleton 002:Hip X", "Skeleton 002:Hip Y", "Skeleton 002:Hip Z", ...]
    # or some variation. Let's locate columns for each bone name.

    # We'll build a lookup that maps (bone_name.lower(), 'x'/'y'/'z') -> column index
    col_lookup = {}
    for col in df.columns:
        col_lower = col.lower()
        # For instance, if the column is "Skeleton 002:Hip X"
        # we extract the base name "skeleton 002:hip" and the axis "x".
        # The exact string matching depends on how the CSV columns are named.
        if " x" in col_lower:
            base_name = col_lower.replace(" x", "").strip()
            col_lookup[(base_name, 'x')] = col
        elif " y" in col_lower:
            base_name = col_lower.replace(" y", "").strip()
            col_lookup[(base_name, 'y')] = col
        elif " z" in col_lower:
            base_name = col_lower.replace(" z", "").strip()
            col_lookup[(base_name, 'z')] = col

    # Let's assume row i=0 or i=1 is the first real frame of data
    # For demonstration, we read just one row (the first frame).
    # If you have multiple frames, you might want to loop through df.itertuples(), etc.
    if len(df) == 0:
        # fallback
        print("CSV is empty or no data rows found; returning random dummy data.")
        return np.random.rand(17,3)

    first_frame = df.iloc[0]  # get row 0 as a Series

    # For each H36M index in TARGET_JOINTS_ORDERED, gather X/Y/Z from the CSV
    # If multiple bone names exist for that index, average them.
    # This yields a (17,3) array
    keypoints_3d = np.zeros((17,3), dtype=np.float32)
    for j_idx, bone_names in TARGET_JOINTS_ORDERED.items():
        coords_list = []
        for bone_name in bone_names:
            # We'll do a simple lower-case match
            bone_name_lower = bone_name.lower()
            x_col = col_lookup.get((bone_name_lower, 'x'), None)
            y_col = col_lookup.get((bone_name_lower, 'y'), None)
            z_col = col_lookup.get((bone_name_lower, 'z'), None)

            if x_col and y_col and z_col:
                try:
                    x_val = float(first_frame[x_col])
                    y_val = float(first_frame[y_col])
                    z_val = float(first_frame[z_col])
                    coords_list.append([x_val, y_val, z_val])
                except (ValueError, TypeError):
                    pass

        if len(coords_list) == 0:
            # if none found, we fallback to zero
            keypoints_3d[j_idx] = [0.0, 0.0, 0.0]
        else:
            coords_arr = np.array(coords_list)
            mean_xyz = coords_arr.mean(axis=0)
            keypoints_3d[j_idx] = mean_xyz

    return keypoints_3d


def main_demo(csv_file, intrinsic_json, extrinsic_json, video_file):
    """
    Main demonstration function:
    1) Parse CSV to get 3D keypoints in the H3.6M ordering
    2) Load camera intrinsics/extrinsics from JSON
    3) Project 3D -> 2D
    4) Draw them on video frames
    """

    # 1) Parse CSV
    keypoints_3d = parse_optitrack_csv(csv_file)  # shape (17, 3)

    # 2) Load camera parameters
    camera_matrix, dist_coeffs, best_extrinsic = load_camera_parameters_json(
        intrinsic_json, extrinsic_json
    )
    # best_extrinsic is 3x4
    best_extrinsic = best_extrinsic.reshape(3,4)

    # 3) Project
    keypoints_2d = project_points_3d_to_2d(
        keypoints_3d, camera_matrix, dist_coeffs, best_extrinsic
    )

    # 4) Overlay
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Cannot open video:", video_file)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For demonstration, we draw the same 2D points on every frame
        # In practice, you'd parse multiple frames from CSV, or read frame i
        # and match it with row i in your CSV, etc.
        for (x, y) in keypoints_2d:
            center = (int(x), int(y))
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

        cv2.imshow("Projected H36M joints", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage:
    csv_file = "./mapping/Take 2025-01-03 04.43.45 PM - smoothed.csv"        # your truncated CSV
    intrinsic_json = "./mapping/intrinsic.json"      # from your attachments
    extrinsic_json = "./mapping/camera_extrinsics_all_error.json"
    video_file = "./mapping/2025-01-03 16-43-44.mp4"
    
    main_demo(csv_file, intrinsic_json, extrinsic_json, video_file)