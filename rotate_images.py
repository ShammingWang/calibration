import os
from PIL import Image

folder_path = './data/TopRight/extrinsic_images'  # Specify your folder path here
degrees_to_rotate = 180

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        file_path = os.path.join(folder_path, filename)
        
        # Open image
        with Image.open(file_path) as img:
            # Rotate the image by degrees_to_rotate degrees
            rotated_img = img.rotate(degrees_to_rotate)
            
            # Save the rotated image back
            rotated_img.save(file_path)

print(f"All .jpg images have been rotated by {degrees_to_rotate} degrees.")
