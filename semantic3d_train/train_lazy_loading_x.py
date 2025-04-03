import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gc

class PointCloudLazyDataset(Dataset):
    """
    Lazy-loading dataset for pointclouds that only loads data when needed,
    avoiding memory issues with large pointcloud files.
    """
    
    def __init__(self, training_pairs, num_points=4096, transform=None, train=True, train_split=0.8):
        """
        Args:
            training_pairs: DataFrame with columns 'txt' and 'labels' containing file paths
            num_points: Number of points to sample from each pointcloud
            transform: Optional transform to apply to the data
            train: Whether this is the training set (True) or validation set (False)
            train_split: Ratio of data to use for training vs validation
        """
        # IMPORTANT: Only keep rows with BOTH txt and labels files
        self.training_pairs = training_pairs.dropna(subset=['txt', 'labels']).reset_index(drop=True)
        print(f"After filtering for complete pairs: {len(self.training_pairs)} valid samples")
        
        self.num_points = num_points
        self.transform = transform
        
        # Split data into training and validation sets
        n_samples = len(self.training_pairs)
        train_size = int(n_samples * train_split)
        
        if train:
            self.training_pairs = self.training_pairs.iloc[:train_size]
        else:
            self.training_pairs = self.training_pairs.iloc[train_size:]
            
        print(f"{'Training' if train else 'Validation'} set contains {len(self.training_pairs)} samples")
        
        # Create label mapping (without loading all the data)
        self.label_mapping = self._create_label_mapping()
        self.num_classes = len(self.label_mapping)
        print(f"Found {self.num_classes} unique classes")
        
    def _create_label_mapping(self):
        """Create a mapping from original label values to consecutive integers"""
        # Initialize with known classes - this is much faster than scanning all files
        known_classes = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        
        # Sample a few files to check for additional classes
        sample_size = min(5, len(self.training_pairs))
        sample_rows = self.training_pairs.sample(sample_size) if len(self.training_pairs) > 0 else []
        
        for _, row in sample_rows.iterrows():
            try:
                # Only read a small chunk to identify classes
                labels = pd.read_table(row['labels'], sep=' ', nrows=1000, names=['class'], index_col=False)
                known_classes.update(labels['class'].unique())
            except Exception as e:
                print(f"Warning: Could not read labels from {row['labels']}: {e}")
        
        # Create the mapping
        mapping = {int(label): i for i, label in enumerate(sorted(known_classes))}
        print(f"Label mapping: {mapping}")
        return mapping
        
    def __len__(self):
        return len(self.training_pairs)
        
    def __getitem__(self, idx):
        """Lazy-load a single pointcloud sample"""
        row = self.training_pairs.iloc[idx]
        
        try:
            # Read data in chunks to reduce memory usage
            xyz_data = self._load_xyz_file(row['txt'])
            label_data = self._load_label_file(row['labels'])
            
            # Make sure they have the same length
            min_len = min(len(xyz_data), len(label_data))
            xyz_data = xyz_data[:min_len]
            label_data = label_data[:min_len]
            
            # Sample points if needed
            if len(xyz_data) > self.num_points:
                # Random sampling
                indices = np.random.choice(len(xyz_data), self.num_points, replace=False)
                xyz_data = xyz_data[indices]
                label_data = label_data[indices]
            elif len(xyz_data) < self.num_points:
                # If we don't have enough points, duplicate some (with small noise)
                needed = self.num_points - len(xyz_data)
                indices = np.random.choice(len(xyz_data), needed, replace=True)
                
                # Add small noise to duplicated points
                extra_xyz = xyz_data[indices].copy()
                extra_xyz[:, :3] += np.random.normal(0, 0.001, extra_xyz[:, :3].shape)
                
                xyz_data = np.vstack([xyz_data, extra_xyz])
                label_data = np.concatenate([label_data, label_data[indices]])
            
            # Extract points, colors and labels
            points = xyz_data[:, :3]  # x, y, z
            colors = xyz_data[:, 4:7] / 255.0  # r, g, b (normalized)
            labels = np.array([self.label_mapping.get(int(label), 0) for label in label_data])
            
            # Convert to torch tensors
            points_tensor = torch.FloatTensor(points)
            colors_tensor = torch.FloatTensor(colors)
            labels_tensor = torch.LongTensor(labels)
            
            # Apply any transformations
            if self.transform:
                points_tensor, colors_tensor, labels_tensor = self.transform(points_tensor, colors_tensor, labels_tensor)
                
            return points_tensor, colors_tensor, labels_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx} (file: {row['txt']}): {e}")
            # Return a dummy sample with all zeros
            dummy_points = torch.zeros((self.num_points, 3))
            dummy_colors = torch.zeros((self.num_points, 3))
            dummy_labels = torch.zeros(self.num_points, dtype=torch.long)
            return dummy_points, dummy_colors, dummy_labels
    
    def _load_xyz_file(self, file_path, chunk_size=10000):
        """Load XYZ file in chunks to reduce memory usage"""
        try:
            # First check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist")
                return np.zeros((0, 7), dtype=np.float32)
            
            # First count lines to allocate array
            line_count = 0
            with open(file_path, 'r') as f:
                for _ in f:
                    line_count += 1
            
            # Allocate array of appropriate size
            data = np.zeros((line_count, 7), dtype=np.float32)
            
            # Read file in chunks
            start_idx = 0
            with open(file_path, 'r') as f:
                while True:
                    lines = []
                    for _ in range(chunk_size):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    
                    if not lines:
                        break
                    
                    # Parse chunk
                    for i, line in enumerate(lines):
                        values = line.strip().split()
                        if len(values) >= 7:  # Make sure line has enough elements
                            data[start_idx + i, :] = [float(val) for val in values[:7]]
                    
                    start_idx += len(lines)
                    
                    # Clean up
                    del lines
                    gc.collect()
            
            return data[:start_idx]  # Return only filled part
        except Exception as e:
            print(f"Error reading XYZ file {file_path}: {e}")
            return np.zeros((0, 7), dtype=np.float32)
    
    def _load_label_file(self, file_path, chunk_size=10000):
        """Load label file in chunks to reduce memory usage"""
        try:
            # First check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist")
                return np.zeros(0, dtype=np.int32)
            
            # First count lines to allocate array
            line_count = 0
            with open(file_path, 'r') as f:
                for _ in f:
                    line_count += 1
            
            # Allocate array of appropriate size
            data = np.zeros(line_count, dtype=np.int32)
            
            # Read file in chunks
            start_idx = 0
            with open(file_path, 'r') as f:
                while True:
                    lines = []
                    for _ in range(chunk_size):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    
                    if not lines:
                        break
                    
                    # Parse chunk
                    for i, line in enumerate(lines):
                        values = line.strip().split()
                        if values:  # Check if line is not empty
                            data[start_idx + i] = int(values[0])
                    
                    start_idx += len(lines)
                    
                    # Clean up
                    del lines
                    gc.collect()
            
            return data[:start_idx]  # Return only filled part
        except Exception as e:
            print(f"Error reading label file {file_path}: {e}")
            return np.zeros(0, dtype=np.int32)

# Integration with your existing code
def create_data_loaders(batch_size=16, num_points=4096, train_split=0.8):
    """Create data loaders for training and validation"""
    # Walk through the 'stuff' directory and collect paths
    all_paths = [os.path.join(path, file) for path, _, files in os.walk('stuff') 
                for file in files if ('.labels' in file) or ('.txt' in file)]
    
    # Create a DataFrame with file paths
    all_files_df = pd.DataFrame({'path': all_paths})
    all_files_df['basename'] = all_files_df['path'].map(os.path.basename)
    all_files_df['id'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[0])
    all_files_df['ext'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[1][1:])
    
    print("Total files found:", len(all_files_df))
    print("Files with extension 'txt':", len(all_files_df[all_files_df['ext'] == 'txt']))
    print("Files with extension 'labels':", len(all_files_df[all_files_df['ext'] == 'labels']))
    
    # Create training pairs
    all_training_pairs = all_files_df.pivot_table(
        values='path', columns='ext', index=['id'], aggfunc='first'
    ).reset_index()
    
    # Check how many have both txt and labels
    complete_pairs = all_training_pairs.dropna(subset=['txt', 'labels'])
    print(f"Complete training pairs (with both txt and labels): {len(complete_pairs)}")
    
    if len(complete_pairs) == 0:
        # Debug - show a few examples to see what's going wrong
        print("\nExample file paths:")
        for i, row in all_files_df.sample(min(10, len(all_files_df))).iterrows():
            print(f"{row['basename']} -> id: {row['id']}, ext: {row['ext']}")
        
        print("\nExample training pairs after pivot:")
        for i, row in all_training_pairs.head().iterrows():
            print(f"ID: {row['id']}")
            print(f"  txt: {row.get('txt', 'None')}")
            print(f"  labels: {row.get('labels', 'None')}")
    
    # Create lazy loading dataset
    train_dataset = PointCloudLazyDataset(
        all_training_pairs, 
        num_points=num_points, 
        train=True, 
        train_split=train_split
    )
    
    valid_dataset = PointCloudLazyDataset(
        all_training_pairs, 
        num_points=num_points, 
        train=False, 
        train_split=train_split
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Adjust based on your system
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Adjust based on your system
        pin_memory=True,
    )
    
    return train_loader, valid_loader, train_dataset.num_classes

# Example usage with adjusted directory search
def find_matching_pairs():
    """Find all matching .txt and .labels file pairs"""
    # Collect all txt files 
    txt_files = []
    for root, _, files in os.walk('stuff'):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append((os.path.join(root, file), file))
    
    # Collect all labels files
    label_files = []
    for root, _, files in os.walk('stuff'):
        for file in files:
            if file.endswith('.labels'):
                label_files.append((os.path.join(root, file), file))
    
    # Create lookup dictionary for labels
    label_lookup = {}
    for path, filename in label_files:
        # Extract base filename without extension
        basename = os.path.splitext(filename)[0]
        label_lookup[basename] = path
    
    # Match txt files with labels
    matched_pairs = []
    for txt_path, txt_filename in txt_files:
        # Extract base filename without extension
        basename = os.path.splitext(txt_filename)[0]
        if basename in label_lookup:
            matched_pairs.append({
                'id': basename,
                'txt': txt_path,
                'labels': label_lookup[basename]
            })
    
    return pd.DataFrame(matched_pairs)

# Alternative loader function using direct matching
def create_data_loaders_direct(batch_size=16, num_points=4096, train_split=0.8):
    """Create data loaders using direct file matching"""
    # Find all matching pairs
    matched_pairs = find_matching_pairs()
    print(f"Found {len(matched_pairs)} matched pairs using direct file matching")
    
    # Create lazy loading dataset
    train_dataset = PointCloudLazyDataset(
        matched_pairs, 
        num_points=num_points, 
        train=True, 
        train_split=train_split
    )
    
    valid_dataset = PointCloudLazyDataset(
        matched_pairs, 
        num_points=num_points, 
        train=False, 
        train_split=train_split
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Adjust based on your system
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Adjust based on your system
        pin_memory=True,
    )
    
    return train_loader, valid_loader, train_dataset.num_classes

# Example usage in your training loop
def main():
    # Your existing parameters
    NUM_POINTS = 4096
    BATCH_SIZE = 16
    EPOCHS = 1000
    LR = 0.00005
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # First try the original method
    print("Trying original dataloader method...")
    train_loader, valid_loader, num_classes = create_data_loaders(
        batch_size=BATCH_SIZE,
        num_points=NUM_POINTS
    )
    
    # If that doesn't work well, try the direct matching method
    if len(train_loader) <= 1:
        print("\nTrying alternative direct matching method...")
        train_loader, valid_loader, num_classes = create_data_loaders_direct(
            batch_size=BATCH_SIZE,
            num_points=NUM_POINTS
        )
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(valid_loader)}")
    
    # Sample a batch to verify
    for points, colors, labels in train_loader:
        print(f"Batch shapes - Points: {points.shape}, Colors: {colors.shape}, Labels: {labels.shape}")
        break
    
    # Your model setup and training loop here

if __name__ == '__main__':
    main()