import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import open3d as o3d
import torch.utils
from utils import *
from model import PointNetSegHead
import os
import torch.optim as optim
from pointnet_loss import PointNetSegLoss
from torchmetrics.classification import MulticlassMatthewsCorrCoef
import time
import gc
from rich import print
import sys
import open3d as o3
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory




# COLOR_MAP = {
#     0  : [47, 79, 79],    # darkslategray
#     1  : [139, 69, 19],   # saddlebrown
#     2  : [34, 139, 34],   # forestgreen
#     3  : [75, 0, 130],    # indigo
#     4  : [255, 0, 0],     # red
#     5  : [255, 255, 0],   # yellow
#     6  : [0, 255, 0],     # lime
#     7  : [0, 255, 255],   # aqua
#     8  : [0, 0, 255],     # blue
# }

train_split_ratio = 0.8
NUM_POINTS = 4096*2 # train/valid points
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
LR = 0.00005
BATCH_SIZE =16


dir_lis = os.listdir('.')
dir_lis = [d for d in dir_lis if 'trained_models_' in d]
dis_idx_lis = [int(d.split('_')[-1]) for d in dir_lis]
dis_idx_lis.sort()

if len(dis_idx_lis) > 0:
    # tar_idx =dis_idx_lis[-1]
    tar_idx = 4
    model_save_dir = f"trained_models_{dis_idx_lis[-1]}"
else:
    sys.exit()


old_labels = torch.load('reservoir/old_labels.pt')
new_labels = torch.load('reservoir/new_labels.pt')
label_names = torch.load('reservoir/label_names.pt')
NUM_CLASSES = len(old_labels)
# NUM_CLASSES = torch.load('reservoir/num_classes.pt')
COLOR_MAP ={}

for i in range(NUM_CLASSES):
    rgb_color = np.random.randint(0, 255, size=3)
    # print("Mapped label '{}' to {} to [rgb({},{},{})]███[/]".format(
    #     label_names[label], new_labels[i], *rgb_color
    # ))
    COLOR_MAP[i] = rgb_color



# with open(file_path,'r') as file:
#     for _ in range(12):
#         file.readline()

#     points=[]
#     colors=[]

#     for line in file:
#         parts=line.strip().split()
#         if len(parts)==7:
#             x,y,z,intensity,r,g,b=map(float,parts)
#             points.append([x,y,z])
#             colors.append([r/255.0,g/255.0,b/255.0])

# points,colors=np.array(points),np.array(colors)


# file_path = "sg27_station1_intensity_rgb.txt"

# # Read the data from the text file
# data = np.loadtxt(file_path)

# data = data[::100]
# print(data.shape)

# # Separate the data into coordinates (x, y, z) and color (r, g, b)
# points = data[:, :3]  # x, y, z
# colors = data[:, 3:] / 255.0  # Normalize RGB values to [0, 1]


# file_path = "isds_sample_data.pcd"
file_path = "dataset/test.ply"


# Read the PLY file as a point cloud
pcd = o3d.io.read_point_cloud(file_path)

# Check if the file was loaded correctly
if not pcd.is_empty():
    print("Successfully loaded point cloud with", len(pcd.points), "points")
else:
    print("Failed to load point cloud")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

points = np.array(pcd.points)
colors = np.array(pcd.colors)


pcd=o3.geometry.PointCloud()
pcd.points=o3.utility.Vector3dVector(points)
pcd.colors=o3.utility.Vector3dVector(colors)
o3.io.write_point_cloud('reference_scan.ply', pcd)

test_points, test_colors, test_labels, min_test_point, max_test_point = preprocess(points, colors, np.copy(colors[:,0]), npoints = NUM_POINTS, r_prob = 0., split = 'test')
test_dataset = PointDataset(test_points, test_colors, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)



feat_dim = 6
seg_model = PointNetSegHead(feat_dim=feat_dim, num_points = NUM_POINTS, m = NUM_CLASSES)


dir_lis = os.listdir(model_save_dir)
dir_lis = [d for d in dir_lis if 'model_epoch_' in d]
dis_idx_lis = [int(d.split('_')[-1].split('.')[0]) for d in dir_lis]
dis_idx_lis.sort()

if len(dis_idx_lis) > 0:
    model_epoch = dis_idx_lis[-1]

else:
    sys.exit()

model_epoch = 160
seg_model.load_state_dict(torch.load(f"{model_save_dir}/model_epoch_{model_epoch}.pth"))


seg_model = seg_model.to(DEVICE)

mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)


# store best validation iou
best_iou = 0.6
best_mcc = 0.6

# lists to store metrics

valid_loss = []
valid_accuracy = []
valid_mcc = []
valid_iou = []


# manually set alpha weights
alpha = np.ones(NUM_CLASSES)
alpha[1:4] *= 1.0 # balance flat stuff
alpha[-1] *= 1.0  # balance non-flat stuff

gamma = 1
criterion = PointNetSegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE)

val_points = []
pred_class = []
val_colors = []

# get test results after each epoch
with torch.no_grad():

    # place model in evaluation mode
    seg_model = seg_model.eval()

    _valid_loss = []
    _valid_accuracy = []
    _valid_mcc = []
    _valid_iou = []
    for i, (points, targets) in enumerate(test_loader, 0):

        points = points.to(DEVICE)
        targets = targets.squeeze().to(DEVICE)

        val_points.append(points[:,:,:3].cpu()*(max_test_point - min_test_point) + min_test_point)
        val_colors.append(points[:,:,3:].cpu())

        preds, _, A = seg_model(points)
        pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

        pred_class.append(pred_choice.cpu())

        loss = criterion(preds, targets, pred_choice)

        torch.cuda.empty_cache()

        # get metrics
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct/float(BATCH_SIZE*NUM_POINTS)
        mcc = mcc_metric(preds.transpose(2, 1), targets)
        iou = compute_iou(targets, pred_choice)

        # update epoch loss and accuracy
        _valid_loss.append(loss.item())
        _valid_accuracy.append(accuracy)
        _valid_mcc.append(mcc.item())
        _valid_iou.append(iou.item())


    valid_loss.append(np.mean(_valid_loss))
    valid_accuracy.append(np.mean(_valid_accuracy))
    valid_mcc.append(np.mean(_valid_mcc))
    valid_iou.append(np.mean(_valid_iou))


    # pause to cool down
    time.sleep(4)

    torch.cuda.empty_cache()


val_point = torch.cat(val_points)
pred_class = torch.cat(pred_class)
val_colors = torch.cat(val_colors)

val_point = torch.flatten(val_point,start_dim=0,end_dim=-2)
val_colors = torch.flatten(val_colors,start_dim=0,end_dim=-2)
pred_class = torch.flatten(pred_class).numpy().tolist()
colors = np.array([np.array(COLOR_MAP[x]) for x in pred_class])
normalized_colors = colors / 255.0  # Normalize the RGB values by dividing by 255


pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(val_point.numpy())
pcd.colors = o3.utility.Vector3dVector(val_colors.numpy())
o3.io.write_point_cloud(f'test_original.ply', pcd)


pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(val_point.numpy())
pcd.colors = o3.utility.Vector3dVector(normalized_colors)
o3.io.write_point_cloud(f'test_segmented.ply', pcd)
o3d.visualization.draw_geometries([pcd])


