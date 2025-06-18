import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union, Literal
from pathlib import Path
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

class ECGDataset(Dataset):
    """Custom Dataset for loading ECG signals."""
    def __init__(
        self, 
        data_dir: str, 
        labels_file: str, 
        task_type: Literal['classification', 'regression'], 
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.transform = transform
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df['id'] = self.labels_df['id'].astype(str)
        self.labels_df.set_index('id', inplace=True)
        
        # Get ECG files and their corresponding labels
        self.ecg_files = []
        self.labels = []
        
        for filename in self.data_dir.glob('*.npy'):
            # Get ECG ID (filename without extension)
            ecg_id = filename.stem
            
            # Check if label exists for this ECG
            if ecg_id in self.labels_df.index:
                self.ecg_files.append(filename)
                self.labels.append(self.labels_df.loc[ecg_id, 'y'])
    
    def __len__(self) -> int:
        return len(self.ecg_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, float]]:
        ecg_path = self.ecg_files[idx]
        
        # Load ECG signal
        ecg_signal = np.load(ecg_path)
        
        # Convert to tensor and add channel dimension for 1D CNN
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)  # (1, length)
        
        # Apply transform if any (for ECG this could be additional normalization)
        if self.transform:
            ecg_signal = self.transform(ecg_signal)
        
        # Get label
        label = self.labels[idx]
        
        # Convert label to tensor
        if self.task_type == 'classification':
            label = torch.tensor(label, dtype=torch.long)
        else:  # regression
            label = torch.tensor(label, dtype=torch.float32)
        
        return ecg_signal, label

class VoiceDataset(Dataset):
    """Custom Dataset for loading voice signals."""
    def __init__(
        self, 
        data_dir: str, 
        labels_file: str, 
        task_type: Literal['classification', 'regression'], 
        transform=None,
        target_length: int = 5000  # Changed to 5000 points
    ):
        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.transform = transform
        self.target_length = target_length
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df['id'] = self.labels_df['id'].astype(str)
        self.labels_df.set_index('id', inplace=True)
        
        # Get voice files and their corresponding labels
        self.voice_files = []
        self.labels = []
        
        for filename in self.data_dir.glob('*.npy'):
            # Get voice ID (filename without extension)
            voice_id = filename.stem
            
            # Check if label exists for this voice
            if voice_id in self.labels_df.index:
                self.voice_files.append(filename)
                self.labels.append(self.labels_df.loc[voice_id, 'y'])
    
    def __len__(self) -> int:
        return len(self.voice_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, float]]:
        voice_path = self.voice_files[idx]
        
        # Load voice signal
        voice_signal = np.load(voice_path)
        
        # Ensure correct length (should already be correct from preprocessing)
        if len(voice_signal) != self.target_length:
            # If somehow the length is wrong, apply uniform sampling
            original_length = len(voice_signal)
            indices = np.linspace(0, original_length - 1, self.target_length, dtype=int)
            voice_signal = voice_signal[indices]
        
        # Convert to tensor and add channel dimension for 1D CNN
        voice_signal = torch.tensor(voice_signal, dtype=torch.float32).unsqueeze(0)  # (1, length)
        
        # Apply transform if any
        if self.transform:
            voice_signal = self.transform(voice_signal)
        
        # Get label
        label = self.labels[idx]
        
        # Convert label to tensor
        if self.task_type == 'classification':
            label = torch.tensor(label, dtype=torch.long)
        else:  # regression
            label = torch.tensor(label, dtype=torch.float32)
        
        return voice_signal, label

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

def get_ecg_transforms() -> Dict[str, torch.nn.Module]:
    """Get ECG-specific transforms (currently identity transforms)."""
    # For ECG data, we typically don't apply the same transforms as images
    # These could be signal-specific augmentations in the future
    identity_transform = torch.nn.Identity()
    
    return {
        'train': identity_transform,
        'val': identity_transform,
        'test': identity_transform
    }

def get_voice_transforms() -> Dict[str, torch.nn.Module]:
    """Get voice-specific transforms."""
    # For voice data, we can add signal-specific augmentations
    # For now, we'll use identity transforms
    identity_transform = torch.nn.Identity()
    
    return {
        'train': identity_transform,
        'val': identity_transform,
        'test': identity_transform
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

def create_ecg_data_loaders(
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
    """Create ECG data loaders."""
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create dataset
    dataset = ECGDataset(data_dir, labels_file, task_type)
    
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def create_voice_data_loaders(
    data_dir: str,
    labels_file: str,
    task_type: Literal['classification', 'regression'],
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    random_seed: int = 42,
    target_length: int = 5000  # Changed to 5000 points
) -> Dict[str, DataLoader]:
    """Create train, validation and test data loaders for voice data."""
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create dataset
    dataset = VoiceDataset(
        data_dir,
        labels_file,
        task_type,
        transform=get_voice_transforms()['train'],
        target_length=target_length
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
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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