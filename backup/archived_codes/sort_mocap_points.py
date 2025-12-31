import glob
import os

# -------- User config (edit these) --------

directory_path = ".\\CalibrationInputFolder\\BottomCenter\\mocap_points"

# "auto" detects from folder name; or force: "left" / "right" / "center"
perspective = "auto"

size = (6, 8)

# If None, defaults are chosen based on perspective.
left_right_inversed = True
front_back_inversed = False

Parameter_book = {
    "right": (True, False),
}

# Optional explicit axis mapping override for sorting.
# Set both or neither. Examples:
#   u_axis_spec = "x";  v_axis_spec = "y"
#   u_axis_spec = "-z"; v_axis_spec = "y"
u_axis_spec = None
v_axis_spec = None

def detect_perspective_from_path(path: str) -> str:
    """Best-effort perspective detection from a directory path.

    Looks for path components like: left / right / center (or middle).
    Defaults to "center" if nothing is recognized.
    """

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
    """Parse axis spec like 'x', '-z', 'y' into (sign, axis_char)."""

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
    if axis_spec is None:
        raise ValueError("axis_spec cannot be None")
    sign, axis = axis_spec
    x, y, z = point_xyz
    value = {"x": x, "y": y, "z": z}[axis]
    return sign * value


def _to_uv(point_xyz, perspective: str, u_axis=None, v_axis=None):
    """Project 3D mocap point (x,y,z) to 2D (u,v) for sorting.

    We only need a consistent 'image-like' 2D ordering:
      - v: top -> down
      - u: left -> right

    Assumption (matches current code using y as vertical):
      - v always uses Y.
      - center camera sees X as horizontal.
      - left/right side cameras see Z as horizontal.
        right is mirrored (looking left), so we negate Z.
    """

    if u_axis is not None and v_axis is not None:
        return _axis_value(point_xyz, u_axis), _axis_value(point_xyz, v_axis)

    x, y, z = point_xyz
    perspective = (perspective or "center").lower()
    if perspective == "left":
        u, v = z, y
    elif perspective == "right":
        u, v = -z, y
    else:  # "center"
        u, v = x, y
    return u, v




def sort_points_in_file(
    filename,
    perspective_name: str = "center",
    x_flipped: bool = False,
    front_back_flipped: bool = False,
    u_axis=None,
    v_axis=None,
):
    """
    Reads 3D points (x y z) from 'filename', sorts them in row-major
    order starting from the top-left corner, and writes them back to
    the same file in sorted order. Supports optional horizontal or
    front/back inversions to match mocap capture orientation.
    """

    # --- Step 1: Read points from file ---
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines if any
                continue
            # Parse x, y, z as floats
            x_str, y_str, z_str = line.split()
            x, y, z = float(x_str), float(y_str), float(z_str)
            points.append((x, y, z))

    expected_n = size[0] * size[1]
    if len(points) != expected_n:
        print(f"Warning: {filename} has {len(points)} points (expected {expected_n}).")

    # Sort by what the camera 'sees': top -> down, then left -> right
    points.sort(
        key=lambda p: (
            -_to_uv(p, perspective_name, u_axis=u_axis, v_axis=v_axis)[1],
            _to_uv(p, perspective_name, u_axis=u_axis, v_axis=v_axis)[0],
        )
    )

    if x_flipped or front_back_flipped:
        grid = []
        for i in range(size[0]):
            start = i * size[1]
            end = start + size[1]
            grid.append(points[start:end])

        # Within each row, enforce left -> right ordering again (helps with noise)
        for i in range(len(grid)):
            grid[i].sort(key=lambda p: _to_uv(p, perspective_name, u_axis=u_axis, v_axis=v_axis)[0])

        if x_flipped:
            grid = [list(reversed(row)) for row in grid]

        if front_back_flipped:
            grid = list(reversed(grid))

        points = [point for row in grid for point in row]

    # --- Step 3: Write sorted points back to the same file ---
    with open(filename, 'w') as f:
        for (x, y, z) in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def sort_all_txt_in_directory(
    directory,
    perspective_name: str = "auto",
    x_flipped=None,
    front_back_flipped=None,
    u_axis=None,
    v_axis=None,
):
    """
    Finds all '.txt' files in the specified directory,
    and sorts each one using 'sort_points_in_file'.
    """
    # Build a search pattern (e.g., '/path/to/dir/*.txt')
    pattern = os.path.join(directory, "*.txt")
    
    detected = (
        detect_perspective_from_path(directory)
        if (perspective_name is None or str(perspective_name).lower() == "auto")
        else str(perspective_name).lower()
    )

    # Defaults chosen to match what each camera 'sees'.
    # If your dataset is mirrored differently, override via left_right_inversed/front_back_inversed.
    default_flips = {
        "left": (False, False),
        "right": (True, False),
        "center": (False, False),
    }
    default_x_flip, default_fb_flip = default_flips.get(detected, (False, False))
    resolved_x_flip = _resolve_optional_bool(x_flipped, default_x_flip)
    resolved_fb_flip = _resolve_optional_bool(front_back_flipped, default_fb_flip)

    print(
        f"Perspective: {detected} | left_right_inversed={resolved_x_flip} | front_back_inversed={resolved_fb_flip}"
    )

    # Loop over all matching .txt files
    for txt_file in glob.glob(pattern):
        print(f"Sorting {txt_file} ...")
        sort_points_in_file(
            txt_file,
            detected,
            resolved_x_flip,
            resolved_fb_flip,
            u_axis=u_axis,
            v_axis=v_axis,
        )
    print("All .txt files in the directory have been sorted.")
    
def rearrange_lines(lines, order_string):
    """
    Reorders 'lines' according to the positions in 'order_string'.

    :param lines: list of strings (each representing one line)
    :param order_string: e.g. "3 2 1 13 14"
    :return: new list of lines in that order

    For example, if order_string = "3 2 1 13",
    then the new order is:
        - the original 3rd line -> new first
        - the original 2nd line -> new second
        - the original 1st line -> new third
        - the original 13th line -> new fourth
    """
    # 1) Parse the order string into a list of integers (1-based indices)
    order_indices = [int(x) for x in order_string.split()]

    # 2) Build the reordered list, adjusting for 0-based indexing in Python
    reordered = []
    for idx in order_indices:
        # 'idx' is 1-based (i.e. "3" means lines[2] in 0-based)
        # Make sure we don't go out of range
        if 1 <= idx <= len(lines):
            reordered.append(lines[idx - 1])
        else:
            # If the index is out of range, decide how to handle it
            # e.g., skip or raise an error. Here we'll just skip.
            print(f"Warning: requested line {idx} is out of range.")
    
    return reordered
    
def rearrange_file_lines(input_file, order_string):
    """
    Reads 'input_file', rearranges its lines according to 'order_string',
    then overwrites 'input_file' with the new order.
    """
    # 1) Read the file lines
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 2) Reorder the lines
    reordered = rearrange_lines(lines, order_string)

    # 3) Write them back
    with open(input_file, 'w') as f:
        for line in reordered:
            f.write(line)


if __name__ == "__main__":

    u_axis = _parse_axis(u_axis_spec) if u_axis_spec else None
    v_axis = _parse_axis(v_axis_spec) if v_axis_spec else None
    if (u_axis is None) ^ (v_axis is None):
        raise SystemExit("Set both u_axis_spec and v_axis_spec (or neither).")

    sort_all_txt_in_directory(
        directory_path,
        perspective,
        left_right_inversed,
        front_back_inversed,
        u_axis=u_axis,
        v_axis=v_axis,
    )