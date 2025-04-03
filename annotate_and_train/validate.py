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

# Walking through the 'stuff' directory and printing all file paths
for dirname, _, filenames in os.walk('stuff'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  # Print full paths for debug

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# Collect paths of '.labels' and '.txt' files in the 'stuff' directory
all_paths = [os.path.join(path, file) for path, _, files in os.walk('stuff') 
             for file in files if ('.labels' in file) or ('.txt' in file)]




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




# metadata = torch.load('dataset_metadata.pt')
# train_dataset = HDF5PointDataset(metadata['train_dataset_path'], split=metadata['train_split'])
# test_dataset = HDF5PointDataset(metadata['test_dataset_path'], split=metadata['test_split'])
train_dataset = torch.load('reservoir/train_dataset.pt')
test_dataset = torch.load('reservoir/test_dataset.pt')
num_train = torch.load('reservoir/num_train.pt')
num_test = torch.load('reservoir/num_test.pt')
min_train_point = torch.load('reservoir/min_train_point.pt')
max_train_point = torch.load('reservoir/max_train_point.pt')
min_test_point = torch.load('reservoir/min_test_point.pt')
max_test_point = torch.load('reservoir/max_test_point.pt')
NUM_CLASSES = torch.load('reservoir/num_classes.pt')
# old_labels = torch.load('reservoir/old_labels.pt')

COLOR_MAP ={}

for i in range(NUM_CLASSES):
    rgb_color = np.random.randint(0, 255, size=3)
    # print("Mapped label '{}' to {} to [rgb({},{},{})]███[/]".format(
    #     label_names[label], new_labels[i], *rgb_color
    # ))
    COLOR_MAP[i] = rgb_color


print("Training dataset size: ", len(train_dataset))
print("Testing dataset size: ", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

points, targets = next(iter(train_loader))
print(f"Input Shape: {points.shape}")
feat_dim = points.shape[-1]
seg_model = PointNetSegHead(feat_dim=feat_dim, num_points = NUM_POINTS, m = NUM_CLASSES)


dir_lis = os.listdir(model_save_dir)
dir_lis = [d for d in dir_lis if 'model_epoch_' in d]
dis_idx_lis = [int(d.split('_')[-1].split('.')[0]) for d in dir_lis]
dis_idx_lis.sort()

if len(dis_idx_lis) > 0:
    model_epoch = dis_idx_lis[-1]

else:
    sys.exit()


seg_model.load_state_dict(torch.load(f"{model_save_dir}/model_epoch_{model_epoch}.pth"))
out, _, _ = seg_model(points)
print(f'Seg shape: {out.shape}')



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

# stuff for training
num_valid_batch = int(np.ceil(num_test/BATCH_SIZE))
# manually set alpha weights
alpha = np.ones(NUM_CLASSES)
alpha[1:4] *= 1.0 # balance flat stuff
alpha[-1] *= 1.0  # balance non-flat stuff

gamma = 1
criterion = PointNetSegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE)

val_points = []
pred_class = []
val_colors = []
ref_class = []


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
        ref_class.append(targets.cpu())

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
ref_class = torch.cat(ref_class)

val_point = torch.flatten(val_point,start_dim=0,end_dim=-2)
val_colors = torch.flatten(val_colors,start_dim=0,end_dim=-2)
pred_class = torch.flatten(pred_class).numpy().tolist()
ref_class = torch.flatten(ref_class).numpy().tolist()


colors = np.array([np.array(COLOR_MAP[x]) for x in pred_class])
normalized_colors = colors / 255.0  # Normalize the RGB values by dividing by 255

val_colors = np.array([np.array(COLOR_MAP[x]) for x in ref_class])
normalized_val_colors = val_colors / 255.0

pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(val_point.numpy())
pcd.colors = o3.utility.Vector3dVector(normalized_val_colors)
o3.io.write_point_cloud(f'val_original.ply', pcd)
o3d.visualization.draw_geometries([pcd])


pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(val_point.numpy())
pcd.colors = o3.utility.Vector3dVector(normalized_colors)
o3.io.write_point_cloud(f'val_segmented.ply', pcd)
o3d.visualization.draw_geometries([pcd])



