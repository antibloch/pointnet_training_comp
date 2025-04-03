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





train_split_ratio = 0.8
NUM_POINTS = 4096*2 # train/valid points
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500
LR = 0.00005
BATCH_SIZE =16


# divide the points, colors and labels into chunks 


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([pcd])

# fig = plt.figure(figsize=(15,10))
# ax = plt.axes(projection='3d')
# ax.scatter(
#             new_df['x'], new_df['y'], new_df['z'],
#             c=new_df[['r', 'g', 'b']].values/255, s=3)  
# ax.view_init(15, 165)

# plt.show()


if not os.path.exists('reservoir'):
    os.mkdir('reservoir')

    ply_files = os.listdir("dataset")
    ply_files = [f for f in ply_files if not f.startswith('test')]
    labels = []
    label_names = []
    all_points = []
    all_colors = []

    for ply_file in ply_files:
        label_class = ply_file.split(".")[0].split("_")[-1]
        if label_class not in label_names:
            label_names.append(label_class)

        print(f"Class: {label_class} mapped to {label_names.index(label_class)}")
        pcd = o3d.io.read_point_cloud(os.path.join("dataset", ply_file))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        all_points.append(points)
        all_colors.append(colors)
        label_val = label_names.index(label_class)
        labels.append(label_val*np.ones((points.shape[0], 1)))


    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    labels = np.concatenate(labels, axis=0)


    NUM_CLASSES = len(np.unique(labels))

    indices = np.arange(len(points))
    # np.random.shuffle(indices)

    train_indices = indices[:int(len(points)*train_split_ratio)]
    test_indices = indices[int(len(points)*train_split_ratio):]

    train_points = points[train_indices]
    train_colors = colors[train_indices]
    train_labels = labels[train_indices]

    test_points = points[test_indices]
    test_colors = colors[test_indices]
    test_labels = labels[test_indices]


    train_points, train_colors, train_labels, min_train_point, max_train_point = preprocess(train_points, train_colors, train_labels, npoints = NUM_POINTS, r_prob = 0.25, split = 'train')
    test_points, test_colors, test_labels, min_test_point, max_test_point = preprocess(test_points, test_colors, test_labels, npoints = NUM_POINTS, r_prob = 0., split = 'test')


    print(f"Train Points Shape: {train_points.shape}")
    print(f"Train Colors Shape: {train_colors.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")

    print(f"Test Points Shape: {test_points.shape}")
    print(f"Test Colors Shape: {test_colors.shape}")
    print(f"Test Labels Shape: {test_labels.shape}")

    num_train= len(train_points)
    num_test = len(test_points)


    train_dataset = PointDataset(train_points, train_colors, train_labels)
    test_dataset = PointDataset(test_points, test_colors, test_labels)


    # save the dataset
    torch.save(train_dataset, 'reservoir/train_dataset.pt')
    torch.save(test_dataset, 'reservoir/test_dataset.pt')
    torch.save(num_train, 'reservoir/num_train.pt')
    torch.save(num_test, 'reservoir/num_test.pt')
    torch.save(min_train_point, 'reservoir/min_train_point.pt')
    torch.save(max_train_point, 'reservoir/max_train_point.pt')
    torch.save(min_test_point, 'reservoir/min_test_point.pt')
    torch.save(max_test_point, 'reservoir/max_test_point.pt')
    torch.save(NUM_CLASSES, 'reservoir/num_classes.pt')

#     del train_points, train_colors, train_labels, test_points, test_colors, test_labels
#     del points, colors, labels

#     gc.collect()

else:
    train_dataset = torch.load('reservoir/train_dataset.pt')
    test_dataset = torch.load('reservoir/test_dataset.pt')
    num_train = torch.load('reservoir/num_train.pt')
    num_test = torch.load('reservoir/num_test.pt')
    min_train_point = torch.load('reservoir/min_train_point.pt')
    max_train_point = torch.load('reservoir/max_train_point.pt')
    min_test_point = torch.load('reservoir/min_test_point.pt')
    max_test_point = torch.load('reservoir/max_test_point.pt')
    NUM_CLASSES = torch.load('reservoir/num_classes.pt')



print("Training dataset size: ", len(train_dataset))
print("Testing dataset size: ", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

points, targets = next(iter(train_loader))
print(f"Input Shape: {points.shape}")
feat_dim = points.shape[-1]
seg_model = PointNetSegHead(feat_dim=feat_dim, num_points = NUM_POINTS, m = NUM_CLASSES).to(DEVICE)
# out, _, _ = seg_model(points.to(DEVICE))
# print(f'Seg shape: {out.shape}')



# use inverse class weighting
# alpha = 1 / class_bins
# alpha = (alpha/alpha.max())

# manually set alpha weights
alpha = np.ones(NUM_CLASSES)
# alpha[1:4] *= 1.0 # balance flat stuff
# alpha[-1] *= 1.0  # balance non-flat stuff

gamma = 1

criterion = PointNetSegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE)
optimizer = optim.Adam(seg_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6, last_epoch=-1)



seg_model = seg_model.to(DEVICE)

mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)


# store best validation iou
best_iou = 0.6
best_mcc = 0.6

# lists to store metrics
train_loss = []
train_accuracy = []
train_mcc = []
train_iou = []
valid_loss = []
valid_accuracy = []
valid_mcc = []
valid_iou = []

# stuff for training
num_train_batch = int(np.ceil(num_train/BATCH_SIZE))
num_valid_batch = int(np.ceil(num_test/BATCH_SIZE))



dir_lis = os.listdir('.')
dir_lis = [d for d in dir_lis if 'trained_models_' in d]
dis_idx_lis = [int(d.split('_')[-1]) for d in dir_lis]
dis_idx_lis.sort()

if len(dis_idx_lis) > 0:
    model_save_dir = f"trained_models_{dis_idx_lis[-1] + 1}"

else:
    model_save_dir = f"trained_models_0"

if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

    
for epoch in range(1, EPOCHS + 1):
    # place model in training mode
    seg_model = seg_model.train()
    _train_loss = []
    _train_accuracy = []
    _train_mcc = []
    _train_iou = []
    for i, (points, targets) in enumerate(train_loader, 0):

        points = points.to(DEVICE)
        targets = targets.squeeze().to(DEVICE)

        # zero gradients
        optimizer.zero_grad()

        # get predicted class logits
        preds, _, _ = seg_model(points)

        # get class predictions
        pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

        # get loss and perform backprop
        loss = criterion(preds, targets, pred_choice)
        loss.backward()
        optimizer.step()
        scheduler.step() # update learning rate

        torch.cuda.empty_cache()

        # get metrics
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct/float(BATCH_SIZE*NUM_POINTS)
        mcc = mcc_metric(preds.transpose(2, 1), targets)
        iou = compute_iou(targets, pred_choice)

        # update epoch loss and accuracy
        _train_loss.append(loss.item())
        _train_accuracy.append(accuracy)
        _train_mcc.append(mcc.item())
        _train_iou.append(iou.item())

        if i % 1 == 0:
            print(f'\t [{epoch}: {i}/{num_train_batch}] ' \
                  + f'train loss: {loss.item():.4f} ' \
                  + f'accuracy: {accuracy:.4f} ' \
                  + f'mcc: {mcc:.4f} ' \
                  + f'iou: {iou:.4f}')

    train_loss.append(np.mean(_train_loss))
    train_accuracy.append(np.mean(_train_accuracy))
    train_mcc.append(np.mean(_train_mcc))
    train_iou.append(np.mean(_train_iou))

    print(f'Epoch: {epoch} - Train Loss: {train_loss[-1]:.4f} ' \
          + f'- Train Accuracy: {train_accuracy[-1]:.4f} ' \
          + f'- Train MCC: {train_mcc[-1]:.4f} ' \
          + f'- Train IOU: {train_iou[-1]:.4f}')

    # pause to cool down
    time.sleep(4)


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

            preds, _, A = seg_model(points)
            pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

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

            if i % 1 == 0:
                print(f'\t [{epoch}: {i}/{num_valid_batch}] ' \
                + f'valid loss: {loss.item():.4f} ' \
                + f'accuracy: {accuracy:.4f} '
                + f'mcc: {mcc:.4f} ' \
                + f'iou: {iou:.4f}')

        valid_loss.append(np.mean(_valid_loss))
        valid_accuracy.append(np.mean(_valid_accuracy))
        valid_mcc.append(np.mean(_valid_mcc))
        valid_iou.append(np.mean(_valid_iou))
        print(f'Epoch: {epoch} - Valid Loss: {valid_loss[-1]:.4f} ' \
            + f'- Valid Accuracy: {valid_accuracy[-1]:.4f} ' \
            + f'- Valid MCC: {valid_mcc[-1]:.4f} ' \
            + f'- Valid IOU: {valid_iou[-1]:.4f}')


        # pause to cool down
        time.sleep(4)

        torch.cuda.empty_cache()



    if epoch%3==0 or epoch==EPOCHS:

        torch.save(seg_model.state_dict(), f'{model_save_dir}/model_epoch_{epoch}.pth')


    
plt.figure(figsize=(12, 8))
plt.subplot(1,3,1)
plt.plot(train_loss)
plt.plot(valid_loss)
plt.title('Loss')
plt.grid()
plt.legend(['Train', 'Valid'])
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1,3,2)
plt.plot(train_accuracy)
plt.plot(valid_accuracy)
plt.title('Accuracy')
plt.grid()
plt.legend(['Train', 'Valid'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,3,3)
plt.plot(train_iou)
plt.plot(valid_iou)
plt.title('IoU')
plt.grid()
plt.legend(['Train', 'Valid'])
plt.xlabel('Epochs')
plt.ylabel('IoU')

plt.tight_layout()
plt.savefig(f'{model_save_dir}/loss_acc_iou.png')

if os.path.exists('score_data'):
    os.sys('rm -rf score_data')

os.mkdir(f'{model_save_dir}/score_data')
torch.save(train_loss, f'{model_save_dir}/score_data/train_loss.pth')
torch.save(train_accuracy, f'{model_save_dir}/score_data/train_accuracy.pth')
torch.save(train_iou, f'{model_save_dir}/score_data/train_iou.pth')
torch.save(train_mcc, f'{model_save_dir}/score_data/train_mcc.pth')
torch.save(valid_loss, f'{model_save_dir}/score_data/valid_loss.pth')
torch.save(valid_accuracy, f'{model_save_dir}/score_data/valid_accuracy.pth')
torch.save(valid_iou, f'{model_save_dir}/score_data/valid_iou.pth')
torch.save(valid_mcc, f'{model_save_dir}/score_data/valid_mcc.pth')
