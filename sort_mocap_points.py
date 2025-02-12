import glob
import os

directory_path = "./mocap_points"
left_right_inversed = True
size = (6,8)

### manual sort:
MANUAL = False
file_path = "./mocap_points/20250103_104906.txt"
new_order = '8 7 6 5 4 3 1 2 16 15 13 14 12 11 10 9 24 22 23 20 21 19 18 17 32 30 31 29 28 27 26 25 40 38 39 36 37 34 35 33 48 46 47 45 44 43 42 41'


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


# left
# 2810 new_order = '8 7 6 5 4 3 1 2 16 15 13 14 12 11 10 9 24 22 23 19 21 20 18 17 32 29 31 30 28 27 26 25 40 37 39 36 38 34 35 33 48 45 47 46 44 43 42 41'
# 2937 new_order = '8 7 6 5 4 3 2 1 16 15 13 14 12 11 10 9 24 22 23 20 21 19 18 17 32 29 31 30 27 28 26 25 40 38 39 36 37 34 35 33 48 45 47 46 43 44 42 41'
# 3107 new_order = '8 6 7 5 4 3 1 2 15 16 13 14 12 11 10 9 24 22 23 21 20 19 18 17 32 29 31 30 28 27 26 25 40 38 39 36 37 34 35 33 48 45 46 47 44 43 42 41'
# 3238 new_order = '8 7 6 5 4 3 2 1 16 15 13 14 12 11 10 9 24 22 23 20 21 19 18 17 32 30 31 29 28 27 26 25 40 38 39 36 37 34 35 33 48 47 46 45 44 43 42 41'
# 3416 new_order = '8 6 7 5 4 3 2 1 16 15 13 14 12 11 10 9 24 22 23 20 21 18 19 17 32 30 29 31 27 28 26 25 40 38 39 36 37 34 35 33 48 47 45 46 43 44 41 42'
# 3637
# 3813 new_order = '8 7 6 5 4 3 1 2 16 15 14 13 12 11 10 9 24 22 23 20 21 19 18 17 32 30 31 29 28 27 26 25 40 38 39 37 36 34 35 33 48 47 46 45 44 43 42 41'
# 3922 new_order = '8 7 6 5 3 4 1 2 16 15 12 13 14 10 11 9 24 22 23 21 20 19 18 17 32 30 31 29 28 26 27 25 40 39 38 37 36 35 34 33 48 46 47 44 45 43 42 41'
# 4221 new_order = '6 8 7 5 4 3 2 1 16 15 13 14 12 11 9 10 24 21 23 20 22 18 19 17 32 29 30 31 28 27 26 25 40 37 39 36 38 34 35 33 48 45 46 47 43 44 41 42'
# 4337 new_order = '6 8 7 5 4 3 2 1 16 15 13 14 12 11 9 10 24 21 23 20 22 18 19 17 32 29 30 31 28 27 26 25 40 37 39 36 38 34 35 33 48 45 46 47 43 44 41 42'
# 4503
# 4906 new_order = '8 7 6 5 4 3 1 2 16 15 13 14 12 11 10 9 24 22 23 20 21 19 18 17 32 30 31 29 28 27 26 25 40 38 39 36 37 34 35 33 48 46 47 45 44 43 42 41'

if __name__ == "__main__":
    
    if MANUAL:
        rearrange_file_lines(file_path, new_order)
    else:
        # Example usage:  python sort_points.py /path/to/directory
        sort_all_txt_in_directory(directory_path,left_right_inversed)