# This is a repository for RGB camera calibration

The core code is cali_intrinsic.py and cali_extrinsics.py

you will get two json files named intrinsic.json and camera_extrinsics_single_error.json respectively

you can get camera_extrinsics_all_error.json by test_extrinsic.py

# Run step

## step 1

put lots of images **without 3d points** into the calibration_images2 folder for non-plane caliberation

run cali_intrinsic.py to calculate the intrinsic matrix with distortion coeffience and **generate intrinsic.json file**

## step 2

put lots of images **with 3d points** into the calibration_images folder **with the same name txt file into the mocap_points folder**

run cali_extrinsics.py to calculate the extrinsic matrix and test each extrinsic matrix on **its own image** (generate camera_extrinsics_single_error.json file)

run test_extrinsic.py to calculate the extrinsic matrix and test each extrinsic matrix on **all images** (generate camera_extrinsics_all_error.json file)

## step 3

compare the **camera_extrinsics_single_error.json** and **camera_extrinsics_all_error.json** , then find out the best extrinsic matrix (**usually the result in camera_extrinsics_all_error.json**)
