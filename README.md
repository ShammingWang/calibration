
# Calibration Toolkit

This repository contains small scripts for camera calibration and for inspecting/editing mocap point files used during extrinsic calibration.

## Folder structure (common)

Many scripts assume a “dataset root” folder that contains at least:

```
<dataset_root>/
├── mocap_points/
│   ├── <name>.txt
│   └── ...
└── extrinsic_images/
      ├── <name>.jpg (or .png/.jpeg/...)
      └── ...
```

Notes:

- Each `mocap_points/<name>.txt` is whitespace-separated `x y z` per line.
- Image matching is done by filename stem (`<name>`).

## Live mocap viewer/editor

Script: `plot_mocap_points.py`

Purpose:

- Live 3D viewer for a mocap `.txt` file (points are labeled with their row index).
- Auto-refreshes when the currently viewed file changes on disk.
- Lets you navigate adjacent numbered files and apply in-place transforms.
- Shows a preview of the matching image from `extrinsic_images/`.

How to run:

```cmd
python plot_mocap_points.py
```

Initial configuration (top of file):

- `file_path`: initial mocap file to open.
- `adjust_sector_size`: the “sector size” used by the reverse operation (default `8`).
- `sort_size`: `(rows, cols)` grid size used for the flip logic (default `(6, 8)`).

UI controls:

- `Prev` / `Next`: switches `_N` index in the filename (e.g., `Foo_4.txt` → `Foo_5.txt`).
- `Reverse sectors (...)`: reverses the order within each sector (every `adjust_sector_size` lines), writing back to the same file.
- `Sort file`: sorts points and writes back to the same file.
- `dataset root` + `Browse`: change to another dataset folder containing `mocap_points/` (+ `extrinsic_images/` for preview).

Sorting controls (right panel):

- `perspective`: `auto/left/right/center`. `auto` infers from folder names (e.g., `left`, `right`, `middle`, `center`).
- `x_flip` / `fb_flip`: optional grid flips after sorting.
   - `x_flip` = reverse each row (left/right inversion).
   - `fb_flip` = reverse row order (front/back inversion).
- `rows` / `cols`: grid dimensions used for flips.
- `u-axis` / `v-axis`: optional axis override for sorting (`none`, `x`, `-x`, `y`, `-y`, `z`, `-z`).
   - Sorting rule: sort by `v` descending (top→down), then by `u` ascending (left→right).

Important:

- `Reverse sectors` and `Sort file` modify the viewed `.txt` in-place. Keep a copy if you need to preserve the original.
- `Browse` uses `tkinter`. If it’s unavailable, paste the dataset path into the `dataset root` box and press Enter.

## Camera calibration (intrinsic + extrinsic)

Script: `main_cali_all.py`

What it does:

- Intrinsic calibration using chessboard images (OpenCV) and writes `intrinsic.json`.
- Extrinsic calibration using `solvePnP` per image and writes `extrinsics.json` (also includes reprojection errors and `best_extrinsic`).

Before running:

- Edit the constants near the top of `main_cali_all.py`:
   - `chessboard_rows`, `chessboard_cols`, `chessboard_length_mm`
   - `input_folder` (your dataset root)

Expected inputs (per current code):

- Images: `<input_folder>/extrinsic_images/*.jpg`
- Mocap points: `<input_folder>/mocap_points/sorted/<image_stem>.txt`

Run:

```cmd
python main_cali_all.py
```

Outputs:

- `<input_folder>/intrinsic.json`
- `<input_folder>/extrinsics.json`
- `<input_folder>/ChessboardCorners/` (if enabled)
- `<input_folder>/undistorted_images/`

## Capture images from a webcam

Script: `cap.py`

- Opens webcam index `0`, shows a live view.
- Press `Q` to save a frame into `calibration_images/`.
- Press `Esc` to quit.

## Rotate images in-place

Script: `rotate_images.py`

- Rotates all `.jpg` images in a configured folder (in-place).
- Edit `folder_path` and `degrees_to_rotate` in the script.

## 3D→2D projection demo (mocap → video)

Script: `map_mocap_points.py`

- Demonstration code that:
   - parses an OptiTrack CSV,
   - maps markers to a Human3.6M-like ordering,
   - loads `intrinsic.json` + `extrinsics.json`,
   - projects 3D points to 2D using OpenCV and overlays them on a video.

It is a demo script: edit the file paths in `__main__` to your local data.

## Dependencies

Minimum commonly used packages:

```cmd
pip install numpy pandas matplotlib opencv-python
```

Optional (only for `rotate_images.py`):

```cmd
pip install pillow
```