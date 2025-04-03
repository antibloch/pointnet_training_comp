import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

def normalize_points(points):
    ''' Perform min/max normalization on points
        Same as:
        (x - min(x))/(max(x) - min(x))
        '''
    min_p = points.min(axis=0)
    max_p = points.max(axis=0)
    points = points - min_p
    points /= (max_p - min_p)

    return points, min_p, max_p


def downsample(points, colors, targets, npoints):
    if len(points) > npoints:
        choice = np.random.choice(len(points), npoints, replace=False)
    else:
        # case when there are less points than the desired number
        choice = np.random.choice(len(points), npoints, replace=True)
    points = points[choice, :] 
    colors = colors[choice, :]
    targets = targets[choice]

    return points, colors, targets


def random_rotate(points):
    ''' randomly rotates point cloud about vertical axis.
        Code is commented out to rotate about all axes
        '''
    # construct a randomly parameterized 3x3 rotation matrix
    phi = np.random.uniform(-np.pi, np.pi)
    theta = np.random.uniform(-np.pi, np.pi)
    psi = np.random.uniform(-np.pi, np.pi)

    rot_x = np.array([
        [1,              0,                 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi) ]])

    rot_y = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,                 1,                0],
        [-np.sin(theta), 0, np.cos(theta)]])

    rot_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi),  0],
        [0,              0,                 1]])

    # rot = np.matmul(rot_x, np.matmul(rot_y, rot_z))
    
    return np.matmul(points, rot_z)





class PointDataset(Dataset):
    def __init__(self, points, colors, labels):
        self.points = points
        self.colors = colors
        self.labels = labels
    

    def __getitem__(self, idx):
        points = self.points[idx]
        colors = self.colors[idx]
        targets = self.labels[idx]  

        # combine colors and points
        # feats = np.column_stack((points, colors))
        feats = np.hstack([points, colors])
        # convert to torch
        feats = torch.from_numpy(feats).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return feats, targets
    

    def __len__(self):
        return len(self.points)
    



def preprocess(points, colors, labels, npoints, r_prob, split):

    # add Gaussian noise to point set if not testing
    if split != 'test':
        # add N(0, 1/100) noise
        points += np.random.normal(0., 0.01, points.shape)

        # add random rotation to the point cloud with probability
        if np.random.uniform(0, 1) > 1 - r_prob:
            points = random_rotate(points)

    # Normalize Point Cloud to (0, 1)
    points, min_point, max_point = normalize_points(points)

    # reshape to (len(points)/n_points, n_points, 3)
    cropped_points = points [:int(len(points)/npoints)*npoints]
    cropped_colors = colors [:int(len(colors)/npoints)*npoints]
    cropped_labels = labels [:int(len(labels)/npoints)*npoints]


    points_reshaped = cropped_points.reshape(int(len(points)/npoints), npoints, 3)
    colors_reshaped = cropped_colors.reshape(int(len(labels)/npoints), npoints, 3)
    labels_reshaped = cropped_labels.reshape(-1, npoints)

    return points_reshaped, colors_reshaped, labels_reshaped, min_point, max_point


def compute_iou(targets, predictions):

    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection

    return intersection / union




class HDF5PointDataset(Dataset):
    def __init__(self, h5_file, split="train"):
        self.h5_file = h5_file
        self.split = split
        
        # Open HDF5 file in read mode
        self.h5f = h5py.File(h5_file, "r")

        # Get references to datasets
        self.points = self.h5f[f"{split}_points"]
        self.colors = self.h5f[f"{split}_colors"]
        self.labels = self.h5f[f"{split}_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Read only required index (lazy loading)
        return (
            torch.cat([torch.tensor(self.points[idx], dtype=torch.float32), torch.tensor(self.colors[idx], dtype=torch.float32)], dim=1),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def close(self):
        self.h5f.close()  # Manually close file when done