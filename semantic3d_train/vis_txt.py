import open3d as o3d
import numpy as np

# Specify the file path
file_path = "sg27_station1_intensity_rgb.txt"

# Read the data from the text file
data = np.loadtxt(file_path)
print("Downsampled data shape:", data.shape)
data = data[::100]
print("Downsampled data shape:", data.shape)
# Separate the data into coordinates (x, y, z) and color (r, g, b)
points = data[:, :3]  # x, y, z
colors = data[:, 3:] / 255.0  # Normalize RGB values to [0, 1]

# Create a point cloud object in Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
