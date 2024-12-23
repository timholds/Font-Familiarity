import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class FontDataset(Dataset):
    """
    Dataset for loading font images from NPZ files.
    """
    def __init__(self, root_dir: str, train: bool = True):
        self.root_dir = root_dir
        mode = 'train' if train else 'test'
        data_file = os.path.join(root_dir, f'{mode}.npz')
        
        # Load data from NPZ file
        with np.load(data_file) as data:
            print("Keys in NPZ file:", data.files)
            self.data = data['images']     # Shape: (N, 256, 256)
            self.targets = data['labels']   # Shape: (N,)
        
        # Load label mapping
        label_map_path = os.path.join(root_dir, 'label_mapping.npy')
        self.label_mapping = np.load(label_map_path, allow_pickle=True).item()
        self.num_classes = len(self.label_mapping)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        target = self.targets[idx]
        
        return img, target

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and test DataLoaders.
    """
    train_dataset = FontDataset(data_dir, train=True)
    test_dataset = FontDataset(data_dir, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.num_classes