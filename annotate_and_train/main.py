import open3d as o3d
import numpy as np
import os


ply_files = os.listdir("ply_files")
labels = []
all_points = []
all_colors = []
label_count = 0
for ply_file in ply_files:
    label_class = ply_file.split(".")[0]
    print(f"Class: {label_class} mapped to {label_count}")
    pcd = o3d.io.read_point_cloud(os.path.join("ply_files", ply_file))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    labels.append(label_count*np.ones((points.shape[0], 1)))
    label_count += 1

all_points = np.concatenate(all_points, axis=0)
all_colors = np.concatenate(all_colors, axis=0)
labels = np.concatenate(labels, axis=0)
