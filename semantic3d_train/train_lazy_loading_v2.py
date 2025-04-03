# Keep all your original imports
import numpy as np
import pandas as pd
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
import h5py
from rich import print

# Keep your parameters as they are
train_split_ratio = 0.8
NUM_POINTS = 4096 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
LR = 0.00005
BATCH_SIZE = 32

# Keep your file path collection code
for dirname, _, filenames in os.walk('stuff'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

all_paths = [os.path.join(path, file) for path, _, files in os.walk('stuff') 
             for file in files if ('.labels' in file) or ('.txt' in file)]

# HDF5PointDataset class definition
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
            torch.cat([
                torch.tensor(self.points[idx], dtype=torch.float32), 
                torch.tensor(self.colors[idx], dtype=torch.float32)
            ], dim=1),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def close(self):
        self.h5f.close()  # Manually close file when done
        
    def __del__(self):
        try:
            self.h5f.close()
        except:
            pass

# Function to check if dataset files exist
def check_dataset_files_exist():
    required_files = [
        os.path.join('reservoir', 'num_train.pt'),
        os.path.join('reservoir', 'num_test.pt'),
        os.path.join('reservoir', 'min_train_point.pt'),
        os.path.join('reservoir', 'max_train_point.pt'),
        os.path.join('reservoir', 'min_test_point.pt'),
        os.path.join('reservoir', 'max_test_point.pt'),
        os.path.join('reservoir', 'old_labels.pt'),
        os.path.join('reservoir', 'new_labels.pt'),
        os.path.join('reservoir', 'label_names.pt'),
        'dataset_metadata.pt',
        'dataset.h5'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
    
    return True

# Check for data files
if check_dataset_files_exist():
    print("Loading existing dataset...")
    # Load existing dataset
    metadata = torch.load('dataset_metadata.pt')
    train_dataset = HDF5PointDataset(metadata['train_dataset_path'], split=metadata['train_split'])
    test_dataset = HDF5PointDataset(metadata['test_dataset_path'], split=metadata['test_split'])
    num_train = torch.load('reservoir/num_train.pt')
    num_test = torch.load('reservoir/num_test.pt')
    min_train_point = torch.load('reservoir/min_train_point.pt')
    max_train_point = torch.load('reservoir/max_train_point.pt')
    min_test_point = torch.load('reservoir/min_test_point.pt')
    max_test_point = torch.load('reservoir/max_test_point.pt')
    old_labels = torch.load('reservoir/old_labels.pt')
    new_labels = torch.load('reservoir/new_labels.pt')
    label_names = torch.load('reservoir/label_names.pt')
    NUM_CLASSES = len(old_labels)
else:
    print("Creating new dataset...")
    # Create directories if they don't exist
    os.makedirs('reservoir', exist_ok=True)
    os.makedirs('point_clouds', exist_ok=True)

    # Your existing label_names dictionary
    label_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation', 
                  4: 'low vegetation', 5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}

    print("Collecting file paths...")
    # Your existing file organization code
    all_files_df = pd.DataFrame({'path': all_paths})
    all_files_df['basename'] = all_files_df['path'].map(os.path.basename)
    all_files_df['id'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[0])
    all_files_df['ext'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[1][1:])

    print(all_files_df.sample(5))

    all_training_pairs = all_files_df.pivot_table(
        values='path', columns='ext', index=['id'], aggfunc='first'
    ).reset_index()
    
    print(f"Found {len(all_training_pairs)} point cloud files")
    
    # Process each point cloud individually
    def process_point_cloud_chunked(row_idx, row, chunk_size=100000):
        """Process a point cloud file in chunks to save memory"""
        print(f"Processing point cloud {row_idx+1}/{len(all_training_pairs.dropna())}: {row['id']}")
        try:
            # Define chunked readers - using pandas with chunksize
            def read_chunks(txt_path, labels_path, chunk_size):
                # Create iterators that read the file in chunks
                txt_iter = pd.read_csv(txt_path, sep=' ', header=None, 
                                    names=['x', 'y', 'z', 'intensity', 'r', 'g', 'b'],
                                    chunksize=chunk_size)
                
                labels_iter = pd.read_csv(labels_path, sep=' ', header=None,
                                        names=['class'],
                                        chunksize=chunk_size)
                
                # Yield chunks together
                for txt_chunk, label_chunk in zip(txt_iter, labels_iter):
                    # Make sure chunks align
                    min_len = min(len(txt_chunk), len(label_chunk))
                    yield txt_chunk.iloc[:min_len], label_chunk.iloc[:min_len]
            
            # Create output HDF5 file
            h5_path = f'point_clouds/{row["id"]}.h5'
            
            with h5py.File(h5_path, 'w') as h5f:
                # Initialize datasets with unknown size
                points_dataset = h5f.create_dataset(
                    'points', shape=(0, 3), 
                    maxshape=(None, 3), 
                    dtype='float32',
                    chunks=(chunk_size, 3)
                )
                
                colors_dataset = h5f.create_dataset(
                    'colors', shape=(0, 3), 
                    maxshape=(None, 3), 
                    dtype='float32',
                    chunks=(chunk_size, 3)
                )
                
                labels_dataset = h5f.create_dataset(
                    'labels', shape=(0,), 
                    maxshape=(None,), 
                    dtype='int32',
                    chunks=(chunk_size,)
                )
                
                # Process chunks
                current_idx = 0
                
                # Progress tracking
                start_time = time.time()
                chunk_count = 0
                
                # Process each chunk
                for xyz_chunk, label_chunk in read_chunks(row['txt'], row['labels'], chunk_size):
                    # Extract data
                    x = xyz_chunk['x'].values
                    y = xyz_chunk['y'].values
                    z = xyz_chunk['z'].values
                    r = xyz_chunk['r'].values/255.0
                    g = xyz_chunk['g'].values/255.0
                    b = xyz_chunk['b'].values/255.0
                    
                    labels = label_chunk['class'].values
                    
                    # Stack data
                    chunk_points = np.column_stack((x, y, z))
                    chunk_colors = np.column_stack((r, g, b))
                    
                    # Resize datasets
                    chunk_size_actual = len(chunk_points)
                    new_size = current_idx + chunk_size_actual
                    
                    points_dataset.resize((new_size, 3))
                    colors_dataset.resize((new_size, 3))
                    labels_dataset.resize((new_size,))
                    
                    # Store data
                    points_dataset[current_idx:new_size] = chunk_points
                    colors_dataset[current_idx:new_size] = chunk_colors
                    labels_dataset[current_idx:new_size] = labels
                    
                    current_idx = new_size
                    chunk_count += 1
                    
                    # Report progress
                    if chunk_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Processed {chunk_count} chunks ({current_idx} points) in {elapsed:.1f} seconds")
                    
                    # Force garbage collection
                    del chunk_points, chunk_colors, labels
                    gc.collect()
                
                # Store metadata
                h5f.attrs['total_points'] = current_idx
                
            print(f"Completed processing {row['id']} with {current_idx} points")
            return h5_path, current_idx
            
        except Exception as e:
            print(f"Error processing {row['id']}: {e}")
            return None, 0
    
    # Process all point clouds
    print("Processing individual point clouds...")
    point_cloud_files = []
    
    for idx, row in all_training_pairs.dropna().iterrows():
        h5_path, total_points = process_point_cloud_chunked(idx, row)
        if h5_path:
            point_cloud_files.append({
                'id': row['id'],
                'path': h5_path,
                'total_points': total_points
            })
    
    # Save point cloud metadata
    point_cloud_df = pd.DataFrame(point_cloud_files)
    point_cloud_df.to_csv('point_clouds/metadata.csv', index=False)
    
    # Create final dataset from preprocessed point clouds
    def create_dataset_from_point_clouds(point_cloud_df, train_split_ratio=0.8, NUM_POINTS=2048):
        """Create final dataset from preprocessed point clouds"""
        print("Collecting unique labels...")
        unique_labels = set()
        
        # First pass: collect all unique labels
        for idx, row in point_cloud_df.iterrows():
            with h5py.File(row['path'], 'r') as h5f:
                # Read a sample of labels to find uniques (for efficiency)
                sample_size = min(100000, h5f.attrs['total_points'])
                if sample_size > 0:
                    label_sample = h5f['labels'][:sample_size]
                    unique_in_file = np.unique(label_sample)
                    unique_labels.update(unique_in_file)
        
        old_labels = np.array(sorted(list(unique_labels)))
        new_labels = np.arange(len(old_labels))
        label_mapping = {old: new for old, new in zip(old_labels, new_labels)}
        
        # Print label mapping
        for i, label in enumerate(old_labels):
            print(f"Mapped label '{label_names.get(label, 'unknown')}' to {new_labels[i]}")
        
        NUM_CLASSES = len(old_labels)
        
        # Lists to hold preprocessed data
        all_train_points = []
        all_train_colors = []
        all_train_labels = []
        all_test_points = []
        all_test_colors = []
        all_test_labels = []
        
        # Process each point cloud
        for idx, row in point_cloud_df.iterrows():
            print(f"Processing file {idx+1}/{len(point_cloud_df)} for dataset: {row['id']}")
            
            with h5py.File(row['path'], 'r') as h5f:
                # Process in reasonable chunks to avoid memory issues
                total_points = h5f.attrs['total_points']
                chunk_size = min(500000, total_points)  # Process 500K points at a time
                
                for offset in range(0, total_points, chunk_size):
                    # Load chunk of point cloud
                    end_idx = min(offset + chunk_size, total_points)
                    chunk_size_actual = end_idx - offset
                    
                    if chunk_size_actual <= 0:
                        continue
                        
                    points = h5f['points'][offset:end_idx]
                    colors = h5f['colors'][offset:end_idx]
                    labels = h5f['labels'][offset:end_idx]
                    
                    # Map labels
                    for old_label, new_label in label_mapping.items():
                        labels[labels == old_label] = new_label
                    
                    # Split into train/test
                    indices = np.arange(len(points))
                    np.random.shuffle(indices)
                    
                    split_idx = int(len(indices) * train_split_ratio)
                    train_indices = indices[:split_idx]
                    test_indices = indices[split_idx:]
                    
                    # Skip if either set is empty
                    if len(train_indices) == 0 or len(test_indices) == 0:
                        continue
                        
                    # Extract train/test data
                    train_points_chunk = points[train_indices]
                    train_colors_chunk = colors[train_indices]
                    train_labels_chunk = labels[train_indices]
                    
                    test_points_chunk = points[test_indices]
                    test_colors_chunk = colors[test_indices]
                    test_labels_chunk = labels[test_indices]
                    
                    # Apply preprocessing
                    train_points_proc, train_colors_proc, train_labels_proc, min_train, max_train = preprocess(
                        train_points_chunk, train_colors_chunk, train_labels_chunk, 
                        npoints=NUM_POINTS, r_prob=0.25, split='train'
                    )
                    
                    test_points_proc, test_colors_proc, test_labels_proc, min_test, max_test = preprocess(
                        test_points_chunk, test_colors_chunk, test_labels_chunk, 
                        npoints=NUM_POINTS, r_prob=0.0, split='test'
                    )
                    
                    # Add to lists
                    all_train_points.append(train_points_proc)
                    all_train_colors.append(train_colors_proc)
                    all_train_labels.append(train_labels_proc)
                    
                    all_test_points.append(test_points_proc)
                    all_test_colors.append(test_colors_proc)
                    all_test_labels.append(test_labels_proc)
                    
                    # Clean up
                    del points, colors, labels
                    del train_points_chunk, train_colors_chunk, train_labels_chunk
                    del test_points_chunk, test_colors_chunk, test_labels_chunk
                    del train_points_proc, train_colors_proc, train_labels_proc
                    del test_points_proc, test_colors_proc, test_labels_proc
                    gc.collect()
        
        # Modify the combining section
        print("Combining all processed data...")
        
        # Create the output HDF5 file first
        with h5py.File("dataset.h5", "w") as f:
            # Initialize empty datasets that we'll fill incrementally
            n_train_samples = sum(len(arr) for arr in all_train_points)
            n_test_samples = sum(len(arr) for arr in all_test_points)
            
            print(f"Creating datasets for {n_train_samples} training and {n_test_samples} test samples")
            
            # Create resizable datasets
            train_points_ds = f.create_dataset("train_points", 
                                              shape=(0, NUM_POINTS, 3),
                                              maxshape=(n_train_samples, NUM_POINTS, 3), 
                                              dtype='float32',
                                              chunks=(min(100, n_train_samples), NUM_POINTS, 3),
                                              compression="gzip")
            
            train_colors_ds = f.create_dataset("train_colors", 
                                              shape=(0, NUM_POINTS, 3),
                                              maxshape=(n_train_samples, NUM_POINTS, 3), 
                                              dtype='float32',
                                              chunks=(min(100, n_train_samples), NUM_POINTS, 3),
                                              compression="gzip")
            
            train_labels_ds = f.create_dataset("train_labels", 
                                              shape=(0, NUM_POINTS),
                                              maxshape=(n_train_samples, NUM_POINTS), 
                                              dtype='int32',
                                              chunks=(min(100, n_train_samples), NUM_POINTS),
                                              compression="gzip")
            
            test_points_ds = f.create_dataset("test_points", 
                                             shape=(0, NUM_POINTS, 3),
                                             maxshape=(n_test_samples, NUM_POINTS, 3), 
                                             dtype='float32',
                                             chunks=(min(100, n_test_samples), NUM_POINTS, 3),
                                             compression="gzip")
            
            test_colors_ds = f.create_dataset("test_colors", 
                                             shape=(0, NUM_POINTS, 3),
                                             maxshape=(n_test_samples, NUM_POINTS, 3), 
                                             dtype='float32',
                                             chunks=(min(100, n_test_samples), NUM_POINTS, 3),
                                             compression="gzip")
            
            test_labels_ds = f.create_dataset("test_labels", 
                                             shape=(0, NUM_POINTS),
                                             maxshape=(n_test_samples, NUM_POINTS), 
                                             dtype='int32',
                                             chunks=(min(100, n_test_samples), NUM_POINTS),
                                             compression="gzip")
            
            # Add training data in batches
            current_train_idx = 0
            for i, train_batch in enumerate(all_train_points):
                batch_size = len(train_batch)
                if batch_size == 0:
                    continue
                    
                print(f"Adding training batch {i+1}/{len(all_train_points)} with {batch_size} samples")
                
                # Resize datasets to accommodate new data
                new_size = current_train_idx + batch_size
                train_points_ds.resize((new_size, NUM_POINTS, 3))
                train_colors_ds.resize((new_size, NUM_POINTS, 3))
                train_labels_ds.resize((new_size, NUM_POINTS))
                
                # Store this batch
                train_points_ds[current_train_idx:new_size] = all_train_points[i]
                train_colors_ds[current_train_idx:new_size] = all_train_colors[i]
                train_labels_ds[current_train_idx:new_size] = all_train_labels[i]
                
                # Free memory
                current_train_idx = new_size
            
            # Add testing data in batches
            current_test_idx = 0
            for i, test_batch in enumerate(all_test_points):
                batch_size = len(test_batch)
                if batch_size == 0:
                    continue
                    
                print(f"Adding testing batch {i+1}/{len(all_test_points)} with {batch_size} samples")
                
                # Resize datasets to accommodate new data
                new_size = current_test_idx + batch_size
                test_points_ds.resize((new_size, NUM_POINTS, 3))
                test_colors_ds.resize((new_size, NUM_POINTS, 3))
                test_labels_ds.resize((new_size, NUM_POINTS))
                
                # Store this batch
                test_points_ds[current_test_idx:new_size] = all_test_points[i]
                test_colors_ds[current_test_idx:new_size] = all_test_colors[i]
                test_labels_ds[current_test_idx:new_size] = all_test_labels[i]
                
                # Free memory
                current_test_idx = new_size
        
            # Calculate global min/max using the stored data - one batch at a time
            min_train_point = np.array([float('inf'), float('inf'), float('inf')])
            max_train_point = np.array([float('-inf'), float('-inf'), float('-inf')])
            min_test_point = np.array([float('inf'), float('inf'), float('inf')])
            max_test_point = np.array([float('-inf'), float('-inf'), float('-inf')])
            
            print("Calculating min/max values...")
            # Calculate for training data
            for i in range(0, current_train_idx, 100):
                end_idx = min(i + 100, current_train_idx)
                points_batch = train_points_ds[i:end_idx].reshape(-1, 3)
                batch_min = points_batch.min(axis=0)
                batch_max = points_batch.max(axis=0)
                min_train_point = np.minimum(min_train_point, batch_min)
                max_train_point = np.maximum(max_train_point, batch_max)
            
            # Calculate for testing data
            for i in range(0, current_test_idx, 100):
                end_idx = min(i + 100, current_test_idx)
                points_batch = test_points_ds[i:end_idx].reshape(-1, 3)
                batch_min = points_batch.min(axis=0)
                batch_max = points_batch.max(axis=0)
                min_test_point = np.minimum(min_test_point, batch_min)
                max_test_point = np.maximum(max_test_point, batch_max)
            
            # Save the dataset metadata
            num_train = current_train_idx
            num_test = current_test_idx
            
            print(f"Final dataset size: {num_train} training, {num_test} testing samples")
        
        # Save metadata
        torch.save(num_train, 'reservoir/num_train.pt')
        torch.save(num_test, 'reservoir/num_test.pt')
        torch.save(min_train_point, 'reservoir/min_train_point.pt')
        torch.save(max_train_point, 'reservoir/max_train_point.pt')
        torch.save(min_test_point, 'reservoir/min_test_point.pt')
        torch.save(max_test_point, 'reservoir/max_test_point.pt')
        torch.save(old_labels, 'reservoir/old_labels.pt')
        torch.save(new_labels, 'reservoir/new_labels.pt')
        torch.save(label_names, 'reservoir/label_names.pt')
        
        # Clean up memory
        for lst in [all_train_points, all_train_colors, all_train_labels, 
                    all_test_points, all_test_colors, all_test_labels]:
            while lst:
                lst.pop(0)  # Remove items one by one to free memory
        gc.collect()
        
        dataset_metadata = {
            'train_dataset_path': 'dataset.h5',
            'test_dataset_path': 'dataset.h5',
            'train_split': 'train',
            'test_split': 'test',
            'num_classes': NUM_CLASSES
        }
        
        torch.save(dataset_metadata, 'dataset_metadata.pt')
        
        return NUM_CLASSES
    
    # Create the dataset
    print("Creating dataset from preprocessed point clouds...")
    NUM_CLASSES = create_dataset_from_point_clouds(
        point_cloud_df, 
        train_split_ratio=train_split_ratio,
        NUM_POINTS=NUM_POINTS
    )
    
    # Create dataset objects
    train_dataset = HDF5PointDataset("dataset.h5", split="train")
    test_dataset = HDF5PointDataset("dataset.h5", split="test")





# The rest of your code remains unchanged
print("Training dataset size: ", len(train_dataset))
print("Testing dataset size: ", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Continue with your model initialization and training code
points, targets = next(iter(train_loader))
print(f"Input Shape: {points.shape}")
feat_dim = points.shape[-1]
del points, targets
seg_model = PointNetSegHead(feat_dim=feat_dim, num_points=NUM_POINTS, m=NUM_CLASSES)

# Use your existing training loop and visualization code...