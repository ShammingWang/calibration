import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Load the sorted data from the file
file_path = "./mocap_points/20250109_103253.txt"  # Replace with your file path if necessary
sorted_data = pd.read_csv(file_path, sep=" ", header=None, names=["x", "y", "z"])

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Iterate through the data to plot each point with its order ID
for idx, row in sorted_data.iterrows():
    x, y, z = row["x"], row["z"], row["y"]
    print(idx, row)
    ax.scatter(x, y, z, color="blue")  # Plot the point
    ax.text(x, y, z, str(idx + 1), fontsize=8, color="red")  # Add the order ID

# Set plot labels and title
# In our setup, facing the LED brings an X axis pointing to the right
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Z Coordinate")
ax.set_zlabel("Y Coordinate")
ax.set_title("3D Plot of Points with Order ID at " + file_path)

plt.gca().invert_xaxis()
plt.show()