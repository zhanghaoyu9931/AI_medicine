import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union, Literal
import json

class MedicalImageDataset(Dataset):
    """Custom Dataset for loading medical images."""
    def __init__(
        self,
        image_dir: str,
        labels_file: str,
        task_type: Literal['classification', 'regression'],
        transform=None
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.task_type = task_type
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df['id'] = self.labels_df['id'].astype(str)
        self.labels_df.set_index('id', inplace=True)
        
        # Get image files and their corresponding labels
        self.image_files = []
        self.labels = []
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Get image ID (filename without extension)
                img_id = os.path.splitext(filename)[0]
                
                # Check if label exists for this image
                if img_id in self.labels_df.index:
                    self.image_files.append(filename)
                    self.labels.append(self.labels_df.loc[img_id, 'y'])
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, float]]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = self.labels[idx]
        
        # Convert label to tensor
        if self.task_type == 'classification':
            label = torch.tensor(label, dtype=torch.long)
        else:  # regression
            label = torch.tensor(label, dtype=torch.float32)
            
        return image, label

def get_transforms() -> Dict[str, transforms.Compose]:
    """Get train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # For grayscale images
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # For grayscale images
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }

def create_data_loaders(
    data_dir: str,
    labels_file: str,
    task_type: Literal['classification', 'regression'],
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    random_seed: int = 42
) -> Dict[str, DataLoader]:
    """Create train, validation and test data loaders."""
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create dataset
    dataset = MedicalImageDataset(
        data_dir,
        labels_file,
        task_type,
        transform=get_transforms()['train']
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def save_dataset_info(
    data_loaders: Dict[str, DataLoader],
    task_type: Literal['classification', 'regression'],
    save_path: str
) -> None:
    """Save dataset information to a JSON file."""
    info = {
        'task_type': task_type,
        'train_size': len(data_loaders['train'].dataset),
        'val_size': len(data_loaders['val'].dataset),
        'test_size': len(data_loaders['test'].dataset),
        'batch_size': data_loaders['train'].batch_size
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4) 