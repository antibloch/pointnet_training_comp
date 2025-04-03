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

class HDF5PointDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, split="train"):
        self.h5_file = h5_file
        self.split = split
        
        # Open HDF5 file in read mode
        self.h5f = h5py.File(h5_file, "r")

        # Get references to datasets
        self.points = self.h5f[f"{split}_points"]
        self.colors = self.h5f[f"{split}_colors"]
        self.labels = self.h5f[f"{split}_labels"]
        
        # Cache dataset length to avoid repeated access
        self._length = len(self.labels)
        
        print(f"Initialized {split} dataset with {self._length} samples")
        
        # Verify dataset structure
        if self._length > 0:
            # Check shapes
            sample_shape = self.points.shape[1:]
            print(f"Point shape: {sample_shape}")
            sample_color_shape = self.colors.shape[1:]
            print(f"Color shape: {sample_color_shape}")
            sample_label_shape = self.labels.shape[1:]
            print(f"Label shape: {sample_label_shape}")
            
            # Check for NaN or invalid data in a sample
            sample_idx = 0
            sample_points = self.points[sample_idx]
            sample_colors = self.colors[sample_idx]
            sample_labels = self.labels[sample_idx]
            
            print(f"Sample points min/max: {np.min(sample_points)}/{np.max(sample_points)}")
            print(f"Sample colors min/max: {np.min(sample_colors)}/{np.max(sample_colors)}")
            print(f"Sample labels min/max/unique: {np.min(sample_labels)}/{np.max(sample_labels)}/{len(np.unique(sample_labels))}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        try:
            # Read only required index (lazy loading)
            points = torch.tensor(self.points[idx], dtype=torch.float32)
            colors = torch.tensor(self.colors[idx], dtype=torch.float32)
            labels = torch.tensor(self.labels[idx], dtype=torch.long)
            
            # Check for NaN values
            if torch.isnan(points).any() or torch.isnan(colors).any():
                print(f"Warning: NaN values found in data at index {idx}")
                # Find the first valid example to return instead
                for i in range(self._length):
                    if i != idx:
                        alt_points = torch.tensor(self.points[i], dtype=torch.float32)
                        alt_colors = torch.tensor(self.colors[i], dtype=torch.float32)
                        alt_labels = torch.tensor(self.labels[i], dtype=torch.long)
                        if not (torch.isnan(alt_points).any() or torch.isnan(alt_colors).any()):
                            return torch.cat([alt_points, alt_colors], dim=1), alt_labels
            
            # Return concatenated points and colors, plus the labels
            return torch.cat([points, colors], dim=1), labels
            
        except Exception as e:
            print(f"Error accessing dataset item {idx}: {e}")
            # Return a placeholder or the first valid item
            if idx != 0 and self._length > 0:
                return self.__getitem__(0)
            else:
                # Create empty tensors as a last resort
                feat_dim = 6  # 3 for points, 3 for colors
                num_points = self.points.shape[1] if self.points.shape[0] > 0 else 4096
                return torch.zeros((num_points, feat_dim)), torch.zeros((num_points,), dtype=torch.long)

    def close(self):
        """Manually close the HDF5 file"""
        if hasattr(self, 'h5f') and self.h5f:
            self.h5f.close()
            self.h5f = None
        
    def __del__(self):
        """Destructor to ensure file is closed when object is deleted"""
        self.close()




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
    # label_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation', 
    #               4: 'low vegetation', 5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}

    label_names = {0: 'background', 1: 'man-made terrain'}



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
    
    # Function to process point clouds in chunks to handle large files
    def process_point_cloud_chunked(row_idx, row, chunk_size=100000):
        """Process a point cloud file in chunks to save memory"""
        print(f"Processing point cloud {row_idx+1}/{len(all_training_pairs.dropna())}: {row['id']}")
        try:
            # Define a more robust chunked reader
            def read_chunks(txt_path, labels_path, chunk_size):
                """
                Read data from both text and label files in chunks
                
                This improved version ensures proper alignment between point data and labels
                """
                # Check if files exist
                if not os.path.exists(txt_path):
                    raise FileNotFoundError(f"Point cloud file not found: {txt_path}")
                if not os.path.exists(labels_path):
                    raise FileNotFoundError(f"Labels file not found: {labels_path}")
                    
                # Get file sizes to make sure they match
                txt_lines = sum(1 for _ in open(txt_path, 'r'))
                label_lines = sum(1 for _ in open(labels_path, 'r'))
                
                if txt_lines != label_lines:
                    print(f"  Warning: Mismatched line counts - {txt_path}: {txt_lines} lines, {labels_path}: {label_lines} lines")
                    # Use the smaller count to avoid index errors
                    max_lines = min(txt_lines, label_lines)
                else:
                    max_lines = txt_lines
                
                # Process in chunks
                for offset in range(0, max_lines, chunk_size):
                    end = min(offset + chunk_size, max_lines)
                    chunk_size_actual = end - offset
                    
                    # Read chunks from both files
                    txt_chunk = pd.read_csv(
                        txt_path, 
                        sep=' ', 
                        header=None,
                        names=['x', 'y', 'z', 'intensity', 'r', 'g', 'b'],
                        skiprows=offset,
                        nrows=chunk_size_actual
                    )
                    
                    label_chunk = pd.read_csv(
                        labels_path, 
                        sep=' ', 
                        header=None,
                        names=['class'],
                        skiprows=offset,
                        nrows=chunk_size_actual
                    )
                    
                    # Verify data is valid
                    if txt_chunk.isna().any().any():
                        print(f"  Warning: Found NaN values in point data, chunk starting at row {offset}")
                        # Clean the data - fill NaNs or drop rows with NaNs
                        txt_chunk = txt_chunk.dropna()
                    
                    if label_chunk.isna().any().any():
                        print(f"  Warning: Found NaN values in labels, chunk starting at row {offset}")
                        # Clean the data - drop rows with NaN labels as we can't guess the correct label
                        valid_indices = ~label_chunk['class'].isna()
                        label_chunk = label_chunk[valid_indices]
                        txt_chunk = txt_chunk[valid_indices]
                    
                    # Skip empty chunks
                    if len(txt_chunk) == 0 or len(label_chunk) == 0:
                        print(f"  Warning: Empty chunk at offset {offset}, skipping")
                        continue
                        
                    # Ensure alignment by index
                    min_len = min(len(txt_chunk), len(label_chunk))
                    yield txt_chunk.iloc[:min_len].reset_index(drop=True), label_chunk.iloc[:min_len].reset_index(drop=True)
            
            # Create output HDF5 file
            h5_path = f'point_clouds/{row["id"]}.h5'
            os.makedirs(os.path.dirname(h5_path), exist_ok=True)
            
            with h5py.File(h5_path, 'w') as h5f:
                # Initialize datasets with unknown size
                points_dataset = h5f.create_dataset(
                    'points', shape=(0, 3), 
                    maxshape=(None, 3), 
                    dtype='float32',
                    chunks=(min(chunk_size, 10000), 3)
                )
                
                colors_dataset = h5f.create_dataset(
                    'colors', shape=(0, 3), 
                    maxshape=(None, 3), 
                    dtype='float32',
                    chunks=(min(chunk_size, 10000), 3)
                )
                
                labels_dataset = h5f.create_dataset(
                    'labels', shape=(0,), 
                    maxshape=(None,), 
                    dtype='int32',
                    chunks=(min(chunk_size, 10000),)
                )
                
                # Process chunks
                current_idx = 0
                
                # Progress tracking
                start_time = time.time()
                chunk_count = 0
                
                # Process each chunk
                for xyz_chunk, label_chunk in read_chunks(row['txt'], row['labels'], chunk_size):
                    try:
                        # Convert to numeric data explicitly to catch any parsing issues
                        for col in ['x', 'y', 'z', 'r', 'g', 'b']:
                            xyz_chunk[col] = pd.to_numeric(xyz_chunk[col], errors='coerce')
                        
                        label_chunk['class'] = pd.to_numeric(label_chunk['class'], errors='coerce').astype(np.int32)
                        
                        # Skip chunk if we have any NaN after conversion
                        if xyz_chunk.isna().any().any() or label_chunk.isna().any().any():
                            print(f"  Warning: Found NaN values after conversion in chunk {chunk_count+1}, skipping")
                            continue
                        
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
                        if chunk_size_actual == 0:
                            continue
                            
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
                        
                    except Exception as chunk_error:
                        print(f"  Error processing chunk {chunk_count+1}: {chunk_error}")
                        continue
                        
                    finally:
                        # Force garbage collection
                        del xyz_chunk, label_chunk
                        gc.collect()
                
                # Store metadata
                h5f.attrs['total_points'] = current_idx
                
            print(f"Completed processing {row['id']} with {current_idx} points")
            return h5_path, current_idx
            
        except Exception as e:
            print(f"Error processing {row['id']}: {e}")
            import traceback
            traceback.print_exc()
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
        
        # If no unique labels found, add a default class (0)
        if not unique_labels:
            print("Warning: No labels found in any point cloud. Adding default class 0.")
            unique_labels.add(0)
        
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
        
        # Get total sample counts
        n_train_samples = sum(len(arr) for arr in all_train_points)
        n_test_samples = sum(len(arr) for arr in all_test_points)
        
        print(f"Collected {n_train_samples} training and {n_test_samples} test samples")
        
        # Handle the case when no samples were collected
        if n_train_samples == 0 or n_test_samples == 0:
            print("Warning: Insufficient data collected to create dataset.")
            print("Creating a minimal placeholder dataset with dummy data...")
            
            # Create placeholder data (1 sample per set with all zeros)
            dummy_train_points = np.zeros((1, NUM_POINTS, 3), dtype=np.float32)
            dummy_train_colors = np.zeros((1, NUM_POINTS, 3), dtype=np.float32)
            dummy_train_labels = np.zeros((1, NUM_POINTS), dtype=np.int32)
            
            dummy_test_points = np.zeros((1, NUM_POINTS, 3), dtype=np.float32)
            dummy_test_colors = np.zeros((1, NUM_POINTS, 3), dtype=np.float32)
            dummy_test_labels = np.zeros((1, NUM_POINTS), dtype=np.int32)
            
            # Use these as the data if real data is missing
            if n_train_samples == 0:
                all_train_points = [dummy_train_points]
                all_train_colors = [dummy_train_colors]
                all_train_labels = [dummy_train_labels]
                n_train_samples = 1
                min_train_point = np.array([0, 0, 0])
                max_train_point = np.array([1, 1, 1])
            
            if n_test_samples == 0:
                all_test_points = [dummy_test_points]
                all_test_colors = [dummy_test_colors]
                all_test_labels = [dummy_test_labels]
                n_test_samples = 1
                min_test_point = np.array([0, 0, 0])
                max_test_point = np.array([1, 1, 1])
        
        # Calculate mins and maxes if we have real data
        if 'min_train_point' not in locals():
            min_train_point = np.array([float('inf'), float('inf'), float('inf')])
            max_train_point = np.array([float('-inf'), float('-inf'), float('-inf')])
            for points in all_train_points:
                batch_min = points.reshape(-1, 3).min(axis=0)
                batch_max = points.reshape(-1, 3).max(axis=0)
                min_train_point = np.minimum(min_train_point, batch_min)
                max_train_point = np.maximum(max_train_point, batch_max)
        
        if 'min_test_point' not in locals():
            min_test_point = np.array([float('inf'), float('inf'), float('inf')])
            max_test_point = np.array([float('-inf'), float('-inf'), float('-inf')])
            for points in all_test_points:
                batch_min = points.reshape(-1, 3).min(axis=0)
                batch_max = points.reshape(-1, 3).max(axis=0)
                min_test_point = np.minimum(min_test_point, batch_min)
                max_test_point = np.maximum(max_test_point, batch_max)
        
        # Modify the combining section
        print(f"Creating dataset with {n_train_samples} training and {n_test_samples} test samples")
        
        # Stack all arrays
        train_points_np = np.vstack(all_train_points)
        train_colors_np = np.vstack(all_train_colors)
        train_labels_np = np.vstack(all_train_labels)
        
        test_points_np = np.vstack(all_test_points)
        test_colors_np = np.vstack(all_test_colors)
        test_labels_np = np.vstack(all_test_labels)
        
        print(f"Final shapes - Train: {train_points_np.shape}, Test: {test_points_np.shape}")
        
        # Create the output HDF5 file
        with h5py.File("dataset.h5", "w") as f:
            # Create datasets with appropriate chunks
            train_chunks = (min(100, n_train_samples), NUM_POINTS, 3)
            test_chunks = (min(100, n_test_samples), NUM_POINTS, 3)
            
            # Create datasets with the stacked arrays
            f.create_dataset("train_points", 
                            data=train_points_np,
                            chunks=train_chunks,
                            compression="gzip")
            
            f.create_dataset("train_colors", 
                            data=train_colors_np,
                            chunks=train_chunks,
                            compression="gzip")
            
            f.create_dataset("train_labels", 
                            data=train_labels_np,
                            chunks=(train_chunks[0], NUM_POINTS),
                            compression="gzip")
            
            f.create_dataset("test_points", 
                            data=test_points_np,
                            chunks=test_chunks,
                            compression="gzip")
            
            f.create_dataset("test_colors", 
                            data=test_colors_np,
                            chunks=test_chunks,
                            compression="gzip")
            
            f.create_dataset("test_labels", 
                            data=test_labels_np,
                            chunks=(test_chunks[0], NUM_POINTS),
                            compression="gzip")
        
        # Save metadata
        torch.save(n_train_samples, 'reservoir/num_train.pt')
        torch.save(n_test_samples, 'reservoir/num_test.pt')
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