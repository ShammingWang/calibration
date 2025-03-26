import glob
import os

directory_path = "./results/middle/mocap_points"
left_right_inversed = True
size = (6,8)

def sort_points_in_file(filename, x_flipped=False):
    """
    Reads 3D points (x y z) from 'filename', sorts them in row-major
    order starting from the top-left corner, and writes them back to
    the same file in sorted order.
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
    
    points.sort(key=lambda p: (-p[1], -p[0]))
    
    if x_flipped:
        new_points = []
        line_index = 0
        for i in range(size[0]):
            line = []
            for j in range(size[1]):
                line.append(points[i*size[1]+j])
            print(line)
            line.reverse()
            new_points.extend(line)
        points = new_points

    # --- Step 3: Write sorted points back to the same file ---
    with open(filename, 'w') as f:
        for (x, y, z) in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def sort_all_txt_in_directory(directory, x_flipped=False):
    """
    Finds all '.txt' files in the specified directory,
    and sorts each one using 'sort_points_in_file'.
    """
    # Build a search pattern (e.g., '/path/to/dir/*.txt')
    pattern = os.path.join(directory, "*.txt")
    
    # Loop over all matching .txt files
    for txt_file in glob.glob(pattern):
        print(f"Sorting {txt_file} ...")
        sort_points_in_file(txt_file, x_flipped)
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

    sort_all_txt_in_directory(directory_path,left_right_inversed)