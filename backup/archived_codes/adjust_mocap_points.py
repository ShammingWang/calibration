import os


# -------- User config (edit these) --------

# Path to a single mocap .txt file to modify
input_file_path = ""

# Reverse the order inside each sector of N lines (default: 8)
sector_size = 8

# If output_file_path is empty/None, overwrite input_file_path
output_file_path = ""


def reverse_each_sector(lines, sector_size: int):
	"""Reverse the order of lines within each consecutive sector.

	Example with sector_size=8:
	  lines[0:8] reversed, then lines[8:16] reversed, ...

	If the final sector has fewer than sector_size lines, it is also reversed.
	"""

	if sector_size <= 0:
		raise ValueError("sector_size must be a positive integer")

	out = []
	for start in range(0, len(lines), sector_size):
		chunk = lines[start : start + sector_size]
		out.extend(reversed(chunk))
	return out


def reverse_sectors_in_file(file_path: str, sector_size: int = 8, output_path: str | None = None):
	"""Reverse the order inside each sector of a text file.

	- Reads all lines from file_path
	- Reverses each consecutive sector of `sector_size` lines
	- Writes result to output_path (or overwrites file_path if output_path is None/empty)
	"""

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


if __name__ == "__main__":
	out_path = output_file_path if output_file_path else None
	written_to = reverse_sectors_in_file(input_file_path, sector_size=sector_size, output_path=out_path)
	print(f"Wrote: {written_to}")
