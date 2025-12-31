import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons, RadioButtons, TextBox
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import re

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

# -------- Live-view options (edit these) --------

# Replace with your file path if necessary
file_path = "CalibrationInputFolder\\BottomRight\\mocap_points\\BottomCenter_1.txt"

# Sector size ("every 8 rows")
adjust_sector_size = 8

# Sorting config (copied from sort_mocap_points.py, configurable via UI)
sort_perspective = "auto"  # auto/left/right/center
sort_size = (6, 8)  # (rows, cols)
sort_left_right_inversed = True  # None => use defaults based on perspective
sort_front_back_inversed = None
sort_u_axis_spec = "none"  # none/x/-x/y/-y/z/-z
sort_v_axis_spec = "none"  # none/x/-x/y/-y/z/-z

# Live plot refresh interval
live_refresh_interval_ms = 300
# UI sizing
ui_bottom = 0.12

# Right-side panel sizing
panel_x0 = 0.72
panel_w = 0.27

def reverse_each_sector(lines, sector_size: int):
    """Reverse the order of lines within each consecutive sector."""

    if sector_size <= 0:
        raise ValueError("sector_size must be a positive integer")

    out = []
    for start in range(0, len(lines), sector_size):
        chunk = lines[start : start + sector_size]
        out.extend(reversed(chunk))
    return out


def reverse_sectors_in_file(file_path: str, sector_size: int = 8, output_path=None):
    """Reverse the order inside each sector of a text file (in-place by default)."""

    if not file_path:
        raise ValueError("file_path is empty")

    file_path = os.path.normpath(file_path)
    if output_path is not None and output_path != "":
        output_path = os.path.normpath(output_path)
    else:
        output_path = None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = reverse_each_sector(lines, sector_size)

    target = output_path or file_path
    with open(target, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return target





def load_points_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=["x", "y", "z"], engine="python")


def apply_display_axes_style(ax):
    """Keep display identical to the original plot_mocap_points view."""

    # In our setup, facing the LED brings an X axis pointing to the right
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Z Coordinate")
    ax.set_zlabel("Y Coordinate")
    ax.invert_xaxis()


def detect_perspective_from_path(path: str) -> str:
    parts = [p for p in os.path.normpath(path).split(os.sep) if p]
    parts_lower = [p.lower() for p in parts]
    aliases = {
        "left": "left",
        "right": "right",
        "center": "center",
        "centre": "center",
        "middle": "center",
        "mid": "center",
    }
    detected = None
    for part in parts_lower:
        if part in aliases:
            detected = aliases[part]
    return detected or "center"


def _resolve_optional_bool(value, default: bool) -> bool:
    return default if value is None else bool(value)


def _parse_axis(axis: str):
    if axis is None:
        return None
    axis = str(axis).strip().lower()
    if not axis:
        return None

    sign = 1
    if axis.startswith("-"):
        sign = -1
        axis = axis[1:].strip()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Invalid axis '{axis}'. Use x, y, z, -x, -y, -z")
    return sign, axis


def _axis_value(point_xyz, axis_spec):
    sign, axis = axis_spec
    x, y, z = point_xyz
    value = {"x": x, "y": y, "z": z}[axis]
    return sign * value


def _to_uv(point_xyz, perspective: str, u_axis=None, v_axis=None):
    if u_axis is not None and v_axis is not None:
        return _axis_value(point_xyz, u_axis), _axis_value(point_xyz, v_axis)

    x, y, z = point_xyz
    perspective = (perspective or "center").lower()
    if perspective == "left":
        u, v = z, y
    elif perspective == "right":
        u, v = -z, y
    else:
        u, v = x, y
    return u, v


def sort_points_in_file(
    filename: str,
    size,
    perspective_name: str = "auto",
    x_flipped=None,
    front_back_flipped=None,
    u_axis=None,
    v_axis=None,
):
    pattern_rows, pattern_cols = size
    points = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x_str, y_str, z_str = line.split()
            points.append((float(x_str), float(y_str), float(z_str)))

    detected = (
        detect_perspective_from_path(filename)
        if (perspective_name is None or str(perspective_name).lower() == "auto")
        else str(perspective_name).lower()
    )

    default_flips = {
        "left": (False, False),
        "right": (True, False),
        "center": (False, False),
    }
    default_x_flip, default_fb_flip = default_flips.get(detected, (False, False))
    resolved_x_flip = _resolve_optional_bool(x_flipped, default_x_flip)
    resolved_fb_flip = _resolve_optional_bool(front_back_flipped, default_fb_flip)

    points.sort(key=lambda p: (-_to_uv(p, detected, u_axis=u_axis, v_axis=v_axis)[1], _to_uv(p, detected, u_axis=u_axis, v_axis=v_axis)[0]))

    if resolved_x_flip or resolved_fb_flip:
        expected_n = pattern_rows * pattern_cols
        if len(points) != expected_n:
            # Still attempt to flip sectors safely
            pass
        grid = []
        for i in range(pattern_rows):
            start = i * pattern_cols
            end = start + pattern_cols
            grid.append(points[start:end])

        for i in range(len(grid)):
            grid[i].sort(key=lambda p: _to_uv(p, detected, u_axis=u_axis, v_axis=v_axis)[0])

        if resolved_x_flip:
            grid = [list(reversed(row)) for row in grid]
        if resolved_fb_flip:
            grid = list(reversed(grid))

        points = [p for row in grid for p in row]

    with open(filename, "w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    return detected, resolved_x_flip, resolved_fb_flip


def draw_points(ax, df: pd.DataFrame, title: str):
    ax.clear()
    for idx, row in df.iterrows():
        x, y, z = row["x"], row["z"], row["y"]
        ax.scatter(x, y, z, color="blue")
        ax.text(x, y, z, str(idx + 1), fontsize=8, color="red")

    ax.set_title(title)
    apply_display_axes_style(ax)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

plt.subplots_adjust(bottom=ui_bottom, right=panel_x0 - 0.02)

# Image preview axis (extrinsic image)
img_ax = fig.add_axes([panel_x0, 0.69, panel_w, 0.27])
img_ax.set_title("extrinsic image", fontsize=9, pad=2)
img_ax.set_xticks([])
img_ax.set_yticks([])
_img_artist = None


def _infer_dataset_root_from_mocap_file(mocap_file_path: str) -> str:
    p = os.path.abspath(os.path.normpath(mocap_file_path))
    parts = p.split(os.sep)
    parts_lower = [x.lower() for x in parts]
    if "mocap_points" in parts_lower:
        idx = len(parts_lower) - 1 - parts_lower[::-1].index("mocap_points")
        if idx - 1 >= 0:
            return os.sep.join(parts[:idx])
    return os.path.dirname(p)


dataset_root_abs = _infer_dataset_root_from_mocap_file(file_path)


def _mocap_dir_from_root(root_abs: str) -> str:
    return os.path.join(root_abs, "mocap_points")


def _extrinsic_dir_from_root(root_abs: str) -> str:
    return os.path.join(root_abs, "extrinsic_images")


def _list_mocap_files(root_abs: str):
    mocap_dir = _mocap_dir_from_root(root_abs)
    if not os.path.isdir(mocap_dir):
        return []
    files = []
    for name in os.listdir(mocap_dir):
        if name.lower().endswith(".txt"):
            files.append(os.path.join(mocap_dir, name))
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files


def _find_matching_extrinsic_image(mocap_file_path: str, root_abs: str):
    images_dir = _extrinsic_dir_from_root(root_abs)
    if not os.path.isdir(images_dir):
        return None

    stem = os.path.splitext(os.path.basename(mocap_file_path))[0]
    preferred_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    for ext in preferred_exts:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate

    # Fallback: any file whose name starts with the same stem
    stem_lower = stem.lower()
    for name in os.listdir(images_dir):
        if os.path.splitext(name)[0].lower().startswith(stem_lower):
            return os.path.join(images_dir, name)
    return None


file_path_abs = os.path.abspath(file_path)
last_mtime = None


def set_current_file(new_path: str, reason: str):
    global file_path_abs, last_mtime
    file_path_abs = os.path.abspath(new_path)
    last_mtime = None
    try_redraw(reason)


def set_dataset_root(new_root: str, reason: str):
    """Set dataset root folder (must contain mocap_points/; extrinsic_images/ optional)."""

    global dataset_root_abs
    if not new_root:
        try_redraw("dataset root is empty")
        return

    root_abs = os.path.abspath(os.path.normpath(str(new_root).strip().strip('"')))
    mocap_dir = _mocap_dir_from_root(root_abs)
    if not os.path.isdir(mocap_dir):
        try_redraw(f"invalid dataset root (missing mocap_points): {os.path.basename(root_abs)}")
        return

    dataset_root_abs = root_abs

    # Keep viewing same basename if possible, otherwise pick first txt.
    desired_name = os.path.basename(file_path_abs)
    candidate = os.path.join(mocap_dir, desired_name)
    if os.path.exists(candidate):
        set_current_file(candidate, reason)
        return

    files = _list_mocap_files(root_abs)
    if not files:
        try_redraw(f"no .txt files in: {mocap_dir}")
        return
    set_current_file(files[0], reason)


def try_get_neighbor_path(current_path: str, delta: int):
    """Return adjacent numbered filename path or (None, message)."""

    base = os.path.basename(current_path)
    directory = os.path.dirname(current_path)

    m = re.match(r"^(.*?)(\d+)(\.[^.]+)$", base)
    if not m:
        return None, f"Cannot parse index from filename: {base}"

    prefix, num_str, suffix = m.groups()
    num = int(num_str)
    candidate_num = num + delta
    if candidate_num < 0:
        return None, f"No {'previous' if delta < 0 else 'next'} file (index would be negative)"

    candidate = os.path.join(directory, f"{prefix}{candidate_num}{suffix}")
    if not os.path.exists(candidate):
        return None, f"File not found: {os.path.basename(candidate)}"
    return candidate, None


def try_redraw(reason: str = "update"):
    global last_mtime
    global _img_artist

    base_name = os.path.basename(file_path_abs)
    try:
        df = load_points_df(file_path_abs)
        title = f"3D Plot of Points with Order ID at {base_name} ({reason})"
        draw_points(ax, df, title)

        # Update extrinsic image preview
        img_path = _find_matching_extrinsic_image(file_path_abs, dataset_root_abs)
        img_ax.clear()
        img_ax.set_title("extrinsic image", fontsize=9, pad=2)
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        if img_path and os.path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                _img_artist = img_ax.imshow(img)
                img_ax.set_title(os.path.basename(img_path), fontsize=8, pad=2)
            except Exception as e:
                _img_artist = None
                img_ax.text(0.5, 0.5, f"Failed to load image\n{type(e).__name__}: {e}", ha="center", va="center")
        else:
            _img_artist = None
            img_ax.text(0.5, 0.5, "No matching image", ha="center", va="center")

        fig.canvas.draw_idle()
        last_mtime = os.path.getmtime(file_path_abs)
    except FileNotFoundError:
        ax.clear()
        ax.set_title(f"File not found: {base_name}")
        apply_display_axes_style(ax)

        img_ax.clear()
        img_ax.set_title("extrinsic image", fontsize=9, pad=2)
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        img_ax.text(0.5, 0.5, "No file", ha="center", va="center")

        fig.canvas.draw_idle()
    except Exception as e:
        ax.clear()
        ax.set_title(f"Failed to read/plot: {base_name}\n{type(e).__name__}: {e}")
        apply_display_axes_style(ax)

        img_ax.clear()
        img_ax.set_title("extrinsic image", fontsize=9, pad=2)
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        img_ax.text(0.5, 0.5, "(plot error)", ha="center", va="center")

        fig.canvas.draw_idle()


def on_timer(_event=None):
    global last_mtime
    try:
        mtime = os.path.getmtime(file_path_abs)
    except FileNotFoundError:
        mtime = None

    if mtime is None:
        if last_mtime is not None:
            try_redraw("missing")
        return

    if last_mtime is None or mtime != last_mtime:
        try_redraw("live")


def on_reverse_clicked(_event=None):
    try:
        reverse_sectors_in_file(file_path_abs, sector_size=adjust_sector_size, output_path=None)
        # mtime resolution can be coarse; force refresh
        time.sleep(0.01)
        try_redraw(f"reversed sectors={adjust_sector_size}")
    except Exception as e:
        ax.set_title(f"Reverse failed: {type(e).__name__}: {e}")
        fig.canvas.draw_idle()


def on_sort_clicked(_event=None):
    global sort_perspective, sort_size, sort_left_right_inversed, sort_front_back_inversed
    global sort_u_axis_spec, sort_v_axis_spec

    try:
        u_spec = str(sort_u_axis_spec).strip().lower()
        v_spec = str(sort_v_axis_spec).strip().lower()
        u_axis = None if u_spec in {"", "none"} else _parse_axis(u_spec)
        v_axis = None if v_spec in {"", "none"} else _parse_axis(v_spec)
        if (u_axis is None) ^ (v_axis is None):
            raise ValueError("Set both u-axis and v-axis (or set both to 'none').")

        detected, xflip, fbflip = sort_points_in_file(
            file_path_abs,
            size=sort_size,
            perspective_name=sort_perspective,
            x_flipped=sort_left_right_inversed,
            front_back_flipped=sort_front_back_inversed,
            u_axis=u_axis,
            v_axis=v_axis,
        )
        try_redraw(f"sorted ({detected}, xflip={xflip}, fbflip={fbflip})")
    except Exception as e:
        try_redraw(f"sort failed: {type(e).__name__}: {e}")


def on_prev_clicked(_event=None):
    candidate, msg = try_get_neighbor_path(file_path_abs, -1)
    if candidate is None:
        try_redraw(msg)
        return
    set_current_file(candidate, "prev")


def on_next_clicked(_event=None):
    candidate, msg = try_get_neighbor_path(file_path_abs, +1)
    if candidate is None:
        try_redraw(msg)
        return
    set_current_file(candidate, "next")


# Buttons
prev_ax = fig.add_axes([0.01, 0.03, 0.08, 0.06])
prev_button = Button(prev_ax, "Prev")
prev_button.on_clicked(on_prev_clicked)

next_ax = fig.add_axes([0.10, 0.03, 0.08, 0.06])
next_button = Button(next_ax, "Next")
next_button.on_clicked(on_next_clicked)

reverse_ax = fig.add_axes([0.19, 0.03, 0.20, 0.06])
reverse_button = Button(reverse_ax, f"Reverse sectors ({adjust_sector_size})")
reverse_button.on_clicked(on_reverse_clicked)

sort_ax = fig.add_axes([0.40, 0.03, 0.10, 0.06])
sort_button = Button(sort_ax, "Sort file")
sort_button.on_clicked(on_sort_clicked)


# Sorting controls (right side)

# Dataset root selector (right side)
root_box_ax = fig.add_axes([panel_x0, 0.64, panel_w - 0.08, 0.045])
root_box_ax.set_title("dataset root", fontsize=9, pad=2)
root_box = TextBox(root_box_ax, "", initial=dataset_root_abs)


def on_root_submit(text):
    set_dataset_root(text, "dataset")


root_box.on_submit(on_root_submit)


def on_browse_clicked(_event=None):
    if tk is None or filedialog is None:
        try_redraw("tkinter not available; type path")
        return
    try:
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(initialdir=dataset_root_abs, title="Select dataset root")
        root.destroy()
    except Exception as e:
        try_redraw(f"browse failed: {type(e).__name__}: {e}")
        return

    if selected:
        root_box.set_val(selected)
        set_dataset_root(selected, "dataset")


browse_ax = fig.add_axes([panel_x0 + panel_w - 0.07, 0.64, 0.07, 0.045])
browse_btn = Button(browse_ax, "Browse")
browse_btn.on_clicked(on_browse_clicked)

persp_ax = fig.add_axes([panel_x0, 0.46, panel_w, 0.17])
persp_radio = RadioButtons(persp_ax, ("auto", "left", "right", "center"), active=0)


def on_perspective_changed(label):
    global sort_perspective
    sort_perspective = str(label)
    try_redraw(f"perspective={sort_perspective}")


persp_radio.on_clicked(on_perspective_changed)
try:
    persp_radio.set_active(("auto", "left", "right", "center").index(sort_perspective))
except Exception:
    pass

flip_ax = fig.add_axes([panel_x0, 0.38, panel_w, 0.07])
_flip_initial = [
    bool(sort_left_right_inversed) if sort_left_right_inversed is not None else False,
    bool(sort_front_back_inversed) if sort_front_back_inversed is not None else False,
]
flip_checks = CheckButtons(flip_ax, ["x_flip", "fb_flip"], _flip_initial)


def _sync_flip_checks_from_globals():
    # CheckButtons has no direct 'set'; toggle if mismatch
    current = flip_checks.get_status()
    desired_x = bool(sort_left_right_inversed) if sort_left_right_inversed is not None else False
    desired_fb = bool(sort_front_back_inversed) if sort_front_back_inversed is not None else False
    if current[0] != desired_x:
        flip_checks.set_active(0)
    if current[1] != desired_fb:
        flip_checks.set_active(1)


def on_flip_toggled(label):
    global sort_left_right_inversed, sort_front_back_inversed
    status = flip_checks.get_status()
    # When user touches the checkbox, treat it as explicit override (not None)
    sort_left_right_inversed = bool(status[0])
    sort_front_back_inversed = bool(status[1])
    try_redraw(f"xflip={sort_left_right_inversed}, fbflip={sort_front_back_inversed}")


flip_checks.on_clicked(on_flip_toggled)
_sync_flip_checks_from_globals()


rows_ax = fig.add_axes([panel_x0, 0.31, panel_w / 2 - 0.01, 0.055])
rows_ax.set_title("rows", fontsize=9, pad=2)
rows_box = TextBox(rows_ax, "", initial=str(sort_size[0]))

cols_ax = fig.add_axes([panel_x0 + panel_w / 2 + 0.01, 0.31, panel_w / 2 - 0.01, 0.055])
cols_ax.set_title("cols", fontsize=9, pad=2)
cols_box = TextBox(cols_ax, "", initial=str(sort_size[1]))


def on_rows_submit(text):
    global sort_size
    sort_size = (int(str(text).strip()), sort_size[1])
    try_redraw(f"size={sort_size[0]}x{sort_size[1]}")


def on_cols_submit(text):
    global sort_size
    sort_size = (sort_size[0], int(str(text).strip()))
    try_redraw(f"size={sort_size[0]}x{sort_size[1]}")


rows_box.on_submit(on_rows_submit)
cols_box.on_submit(on_cols_submit)


axis_options = ("none", "x", "-x", "y", "-y", "z", "-z")

uax_ax = fig.add_axes([panel_x0, 0.12, panel_w / 2 - 0.01, 0.18])
uax_radio = RadioButtons(uax_ax, axis_options, active=0)
uax_ax.set_title("u-axis", fontsize=9, pad=2)

vax_ax = fig.add_axes([panel_x0 + panel_w / 2 + 0.01, 0.12, panel_w / 2 - 0.01, 0.18])
vax_radio = RadioButtons(vax_ax, axis_options, active=0)
vax_ax.set_title("v-axis", fontsize=9, pad=2)


def on_uax_changed(label):
    global sort_u_axis_spec
    sort_u_axis_spec = str(label)
    try_redraw(f"u-axis={sort_u_axis_spec}")


def on_vax_changed(label):
    global sort_v_axis_spec
    sort_v_axis_spec = str(label)
    try_redraw(f"v-axis={sort_v_axis_spec}")


uax_radio.on_clicked(on_uax_changed)
vax_radio.on_clicked(on_vax_changed)

try:
    uax_radio.set_active(axis_options.index(str(sort_u_axis_spec).strip().lower() or "none"))
except Exception:
    pass

try:
    vax_radio.set_active(axis_options.index(str(sort_v_axis_spec).strip().lower() or "none"))
except Exception:
    pass


# Initial draw and live timer
try_redraw("initial")
timer = fig.canvas.new_timer(interval=live_refresh_interval_ms)
timer.add_callback(on_timer)
timer.start()
plt.show()