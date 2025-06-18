import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Literal
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, average_precision_score,
    roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
import seaborn as sns
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MedicalCNN(nn.Module):
    """CNN model for medical image classification/regression."""
    def __init__(
        self,
        task_type: Literal['classification', 'regression'],
        num_classes: int = 2,
        num_conv_layers: int = 4,
        conv_channels: int = 32,
        fc_layers: List[int] = [512, 128]
    ):
        super(MedicalCNN, self).__init__()
        
        # Build convolutional layers
        conv_modules = []
        in_channels = 1  # Grayscale input
        
        for _ in range(num_conv_layers):
            conv_modules.extend([
                nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = conv_channels
        
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # Calculate the size of flattened features
        # Assuming input size is 224x224, after num_conv_layers maxpooling layers
        # the feature map size will be 224/(2^num_conv_layers)
        feature_size = 224 // (2 ** num_conv_layers)
        flattened_size = conv_channels * feature_size * feature_size
        
        # Build fully connected layers
        fc_modules = []
        prev_size = flattened_size
        
        for fc_size in fc_layers:
            fc_modules.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            prev_size = fc_size
        
        # Add final layer based on task type
        if task_type == 'classification':
            fc_modules.append(nn.Linear(fc_layers[-1], num_classes))
        else:  # regression
            fc_modules.append(nn.Linear(fc_layers[-1], 1))
        
        self.fc_layers = nn.Sequential(*fc_modules)
        self.task_type = task_type
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # If input is (batch_size, height, width)
            x = x.unsqueeze(1)  # Add channel dimension -> (batch_size, 1, height, width)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class ECG1DCNN(nn.Module):
    """1D CNN model for ECG signal classification/regression."""
    def __init__(
        self,
        task_type: Literal['classification', 'regression'],
        input_length: int = 50000,
        num_classes: int = 2,
        num_conv_layers: int = 4,
        conv_channels: int = 32,
        kernel_size: int = 7,
        pool_size: int = 2,
        fc_layers: List[int] = [512, 128]
    ):
        super(ECG1DCNN, self).__init__()
        
        # Build 1D convolutional layers
        conv_modules = []
        in_channels = 1  # Single ECG channel
        
        for _ in range(num_conv_layers):
            conv_modules.extend([
                nn.Conv1d(in_channels, conv_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(0.2)
            ])
            in_channels = conv_channels
            # Double channels every second layer
            if len(conv_modules) % 10 == 0:  # Every 2 conv layers (5 modules each)
                conv_channels = min(conv_channels * 2, 512)
        
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # Calculate the size of flattened features
        # After num_conv_layers maxpool operations, length is divided by pool_size^num_conv_layers
        feature_length = input_length // (pool_size ** num_conv_layers)
        flattened_size = conv_channels * feature_length
        
        # Build fully connected layers
        fc_modules = []
        prev_size = flattened_size
        
        for fc_size in fc_layers:
            fc_modules.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            prev_size = fc_size
        
        # Add final layer based on task type
        if task_type == 'classification':
            fc_modules.append(nn.Linear(fc_layers[-1], num_classes))
        else:  # regression
            fc_modules.append(nn.Linear(fc_layers[-1], 1))
        
        self.fc_layers = nn.Sequential(*fc_modules)
        self.task_type = task_type
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # He initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize batch norm parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 1, sequence_length)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class Voice1DCNN(nn.Module):
    """1D CNN model for voice signal classification/regression."""
    def __init__(
        self,
        task_type: Literal['classification', 'regression'],
        input_length: int = 5000,  # Changed to 5000 points
        num_classes: int = 2,
        num_conv_layers: int = 4,
        conv_channels: int = 32,
        kernel_size: int = 7,
        pool_size: int = 2,
        fc_layers: List[int] = [512, 128]
    ):
        super(Voice1DCNN, self).__init__()
        
        # Build 1D convolutional layers optimized for voice signals
        conv_modules = []
        in_channels = 1  # Single voice channel
        
        for i in range(num_conv_layers):
            # Use larger kernels for early layers to capture broader patterns
            current_kernel = kernel_size if i < 2 else max(3, kernel_size // 2)
            
            conv_modules.extend([
                nn.Conv1d(in_channels, conv_channels, kernel_size=current_kernel, 
                         stride=1, padding=current_kernel//2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(0.2)
            ])
            in_channels = conv_channels
            # Increase channels progressively
            if i % 2 == 1:  # Every second layer
                conv_channels = min(conv_channels * 2, 256)
        
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # Calculate the size of flattened features
        # After num_conv_layers maxpool operations, length is divided by pool_size^num_conv_layers
        feature_length = input_length // (pool_size ** num_conv_layers)
        flattened_size = conv_channels * feature_length
        
        # Build fully connected layers
        fc_modules = []
        prev_size = flattened_size
        
        for fc_size in fc_layers:
            fc_modules.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            prev_size = fc_size
        
        # Add final layer based on task type
        if task_type == 'classification':
            fc_modules.append(nn.Linear(fc_layers[-1], num_classes))
        else:  # regression
            fc_modules.append(nn.Linear(fc_layers[-1], 1))
        
        self.fc_layers = nn.Sequential(*fc_modules)
        self.task_type = task_type
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # He initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize batch norm parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 1, sequence_length)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class ModelTrainer:
    """Class to handle model training and evaluation."""
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        task_type: Literal['classification', 'regression'],
        class_names: Optional[Dict[int, str]] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.task_type = task_type
        self.class_names = class_names  # 添加类别名称映射
        self.model.to(device)
        
        # Add learning rate scheduler for better convergence (only if optimizer exists)
        if self.optimizer is not None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        else:
            self.scheduler = None
        
        # Additional initialization optimizations
        self._optimize_for_training()
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []
        
    def _optimize_for_training(self):
        """Additional optimizations for better training convergence."""
        # Set model to training mode for proper initialization
        self.model.train()
        
        # Enable gradient computation for initialization
        for param in self.model.parameters():
            if param.requires_grad:
                # Ensure gradients are properly initialized
                if param.grad is not None:
                    param.grad.zero_()
        
        # Set model back to evaluation mode
        self.model.eval()
        
    def create_optimized_optimizer(self, learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """Create an optimized optimizer with weight decay for better convergence."""
        # Use AdamW optimizer with weight decay for better regularization
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Update the scheduler with the new optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        return optimizer
        
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        return 0.0
        
    def upgrade_to_optimized_optimizer(self, learning_rate: float = None, weight_decay: float = 1e-4):
        """
        Optional method to upgrade the current optimizer to AdamW with weight decay.
        This can be called after creating the trainer to get better convergence.
        
        Args:
            learning_rate: New learning rate (if None, keeps current LR)
            weight_decay: Weight decay for regularization
        """
        if self.optimizer is None:
            print("Warning: No optimizer to upgrade")
            return
            
        # Get current learning rate if not specified
        if learning_rate is None:
            learning_rate = self.get_current_lr()
        
        # Create new AdamW optimizer
        new_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Update scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            new_optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Replace optimizer
        self.optimizer = new_optimizer
        print(f"Upgraded to AdamW optimizer with LR={learning_rate}, weight_decay={weight_decay}")
        
    def set_class_names(self, class_names: Dict[int, str]) -> None:
        """Set class names for classification tasks."""
        self.class_names = class_names
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Add progress bar for training
        train_pbar = tqdm(train_loader, desc='Training', leave=False, 
                         bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
        
        for images, labels in train_pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            if self.task_type == 'regression':
                outputs = outputs.squeeze()
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            current_loss = total_loss / (len(all_preds) // len(images) + 1)
            
            # Store predictions and labels
            if self.task_type == 'classification':
                _, predicted = outputs.max(1)
            else:
                predicted = outputs
            
            # Convert to numpy arrays and ensure they are 1D
            predicted_np = predicted.detach().cpu().numpy().flatten()
            labels_np = labels.detach().cpu().numpy().flatten()
            
            all_preds.extend(predicted_np)
            all_labels.extend(labels_np)
            
            # Update progress bar with current loss
            train_pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        epoch_loss = total_loss / len(train_loader)
        
        return epoch_loss, metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Add progress bar for validation
        val_pbar = tqdm(val_loader, desc='Validation', leave=False, 
                       bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if self.task_type == 'regression':
                    outputs = outputs.squeeze()
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                current_loss = total_loss / (len(all_preds) // len(images) + 1)
                
                # Store predictions and labels
                if self.task_type == 'classification':
                    _, predicted = outputs.max(1)
                else:
                    predicted = outputs
                
                # Convert to numpy arrays and ensure they are 1D
                predicted_np = predicted.detach().cpu().numpy().flatten()
                labels_np = labels.detach().cpu().numpy().flatten()
                
                all_preds.extend(predicted_np)
                all_labels.extend(labels_np)
                
                # Update progress bar with current loss
                val_pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        val_loss = total_loss / len(val_loader)
        
        return val_loss, metrics
    
    def _calculate_metrics(
        self,
        labels: List[Union[int, float]],
        preds: List[Union[int, float]]
    ) -> Dict[str, float]:
        """Calculate metrics based on task type."""
        metrics = {}
        
        # Convert to numpy arrays and ensure they are 1D
        labels_array = np.array(labels).flatten()
        preds_array = np.array(preds).flatten()
        
        # Check if arrays are empty
        if len(labels_array) == 0 or len(preds_array) == 0:
            print("Warning: Empty labels or predictions array")
            return metrics
        
        if self.task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = np.mean(labels_array == preds_array)
            
            # Ensure labels and predictions are integers for sklearn metrics
            labels_int = labels_array.astype(int)
            preds_int = preds_array.astype(int)
            
            try:
                metrics['precision'] = precision_score(labels_int, preds_int, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(labels_int, preds_int, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(labels_int, preds_int, average='weighted', zero_division=0)
                
                # Calculate AUPR and AUC for binary classification
                if len(np.unique(labels_int)) == 2:
                    metrics['aupr'] = average_precision_score(labels_int, preds_int)
                    metrics['auc'] = roc_auc_score(labels_int, preds_int)
            except Exception as e:
                print(f"Warning: Error calculating classification metrics: {e}")
                # Set default values
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0
        else:
            # Regression metrics
            try:
                metrics['mse'] = mean_squared_error(labels_array, preds_array)
                metrics['mae'] = mean_absolute_error(labels_array, preds_array)
                metrics['r2'] = r2_score(labels_array, preds_array)
            except Exception as e:
                print(f"Warning: Error calculating regression metrics: {e}")
                # Set default values
                metrics['mse'] = float('inf')
                metrics['mae'] = float('inf')
                metrics['r2'] = 0.0
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str,
        early_stopping_patience: int = 5
    ) -> Dict[str, List[Union[float, Dict[str, float]]]]:
        """Train the model for specified number of epochs."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Add progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc='Epochs', position=0)
        
        for epoch in epoch_pbar:
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update epoch progress bar with metrics
            if self.task_type == 'classification':
                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Val Loss': f'{val_loss:.4f}',
                    'Train Acc': f'{train_metrics.get("accuracy", 0):.3f}',
                    'Val Acc': f'{val_metrics.get("accuracy", 0):.3f}',
                    'LR': f'{self.get_current_lr():.6f}'
                })
            else:
                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Val Loss': f'{val_loss:.4f}',
                    'Train R²': f'{train_metrics.get("r2", 0):.3f}',
                    'Val R²': f'{val_metrics.get("r2", 0):.3f}',
                    'LR': f'{self.get_current_lr():.6f}'
                })
            
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Metrics: {train_metrics}')
            print(f'Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(save_dir, 'best_model.pth')
                print(f'New best model saved with validation loss: {val_loss:.4f}')
            else:
                patience_counter += 1
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Save training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        self.save_history(history, save_dir)
        
        return history
    
    def save_model(self, save_dir: str, filename: str) -> None:
        """Save model state."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare state dict
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'task_type': self.task_type,
            'class_names': self.class_names
        }
        
        # Add optimizer state if available
        if self.optimizer is not None:
            state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(state_dict, Path(save_dir) / filename)
    
    def load_model(self, model_path: str) -> None:
        """Load model state."""
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available and optimizer exists
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.task_type = checkpoint['task_type']
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
    
    def save_history(self, history: Dict[str, List[Union[float, Dict[str, float]]]], save_dir: str) -> None:
        """Save training history."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=4)
    
    def plot_training_history(self, save_dir: str) -> None:
        """Plot training history."""
        if not self.train_losses or not self.val_losses:
            print("Warning: No training history to plot")
            return
            
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot metrics
        if self.task_type == 'classification':
            metrics = ['accuracy', 'f1']
        else:
            metrics = ['mse', 'r2']
        
        for i, metric in enumerate(metrics, 2):
            plt.subplot(1, 3, i)
            try:
                train_metric = [m.get(metric, 0.0) for m in self.train_metrics]
                val_metric = [m.get(metric, 0.0) for m in self.val_metrics]
                plt.plot(train_metric, label=f'Train {metric.upper()}')
                plt.plot(val_metric, label=f'Val {metric.upper()}')
                # plt.title(f'Training and Validation {metric.upper()}')
                plt.xlabel('Epoch')
                plt.ylabel(metric.upper())
                plt.legend()
            except Exception as e:
                print(f"Warning: Error plotting {metric}: {e}")
                plt.text(0.5, 0.5, f'Error plotting {metric}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'training_history.png')
        plt.close()
    
    def evaluate(self, test_loader: DataLoader, save_dir: str) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # Add progress bar for evaluation
        eval_pbar = tqdm(test_loader, desc='Evaluating', 
                        bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
        
        with torch.no_grad():
            for images, labels in eval_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if self.task_type == 'regression':
                    outputs = outputs.squeeze()
                
                if self.task_type == 'classification':
                    _, predicted = outputs.max(1)
                else:
                    predicted = outputs
                
                # Convert to numpy arrays and ensure they are 1D
                predicted_np = predicted.detach().cpu().numpy().flatten()
                labels_np = labels.detach().cpu().numpy().flatten()
                
                all_preds.extend(predicted_np)
                all_labels.extend(labels_np)
                
                # Update progress
                eval_pbar.set_postfix({'Samples': f'{len(all_preds)}/{len(test_loader.dataset)}'})
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        # Plot confusion matrix for classification
        if self.task_type == 'classification':
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            
            # 如果有类别名称映射，使用具体的类别名称
            if self.class_names:
                labels = [self.class_names[i] for i in sorted(self.class_names.keys())]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels, yticklabels=labels)
            else:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(Path(save_dir) / 'confusion_matrix.png')
            plt.close()
        
        return metrics
    
    def get_predictions_and_probabilities(self, test_loader: DataLoader, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions and probabilities from the model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        if debug:
            print(f"Model is on device: {next(self.model.parameters()).device}")
            print(f"Task type: {self.task_type}")
        
        # Add progress bar for predictions
        pred_pbar = tqdm(test_loader, desc='Getting Predictions', leave=False,
                        bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pred_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                if debug and batch_idx == 0:  # Debug first batch
                    print(f"First batch - Images shape: {images.shape}")
                    print(f"First batch - Labels shape: {labels.shape}")
                    print(f"First batch - Labels sample: {labels[:5]}")
                    print(f"First batch - Outputs shape: {outputs.shape}")
                    print(f"First batch - Outputs sample: {outputs[:2]}")
                
                if self.task_type == 'classification':
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    all_probs.extend(probs.cpu().numpy())  # Keep 2D shape for probabilities
                    all_preds.extend(predicted.detach().cpu().numpy().flatten())
                    
                    if debug and batch_idx == 0:  # Debug first batch
                        print(f"First batch - Probs shape: {probs.shape}")
                        print(f"First batch - Probs sample: {probs[:2]}")
                        print(f"First batch - Predicted sample: {predicted[:5]}")
                else:
                    outputs = outputs.squeeze()
                    all_preds.extend(outputs.cpu().numpy().flatten())
                    all_probs.extend(outputs.cpu().numpy().flatten())  # For regression, probs are the same as predictions
                
                all_labels.extend(labels.cpu().numpy().flatten())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        if debug:
            print(f"Final shapes - Labels: {all_labels.shape}, Preds: {all_preds.shape}, Probs: {all_probs.shape}")
            print(f"Labels unique values: {np.unique(all_labels)}")
            print(f"Predictions unique values: {np.unique(all_preds)}")
        
        return all_labels, all_preds, all_probs

def extract_features_for_ml(model: nn.Module, data_loader: DataLoader, device: torch.device, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from CNN for traditional ML models."""
    model.eval()
    features = []
    labels = []
    
    if debug:
        print(f"Extracting features from model on device: {device}")
    
    # Add progress bar for feature extraction
    feature_pbar = tqdm(data_loader, desc='Extracting Features', leave=False,
                       bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(feature_pbar):
            images = images.to(device)
            
            if debug and batch_idx == 0:  # Debug first batch
                print(f"Feature extraction - Images shape: {images.shape}")
                print(f"Feature extraction - Targets shape: {targets.shape}")
                print(f"Feature extraction - Targets sample: {targets[:5]}")
            
            # Extract features from the CNN (before the final layer)
            x = images
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            # Forward through conv layers
            x = model.conv_layers(x)
            x = x.view(x.size(0), -1)
            
            if debug and batch_idx == 0:  # Debug first batch
                print(f"After conv layers - Features shape: {x.shape}")
            
            # Forward through all but the last FC layer
            for i, layer in enumerate(model.fc_layers[:-1]):  # Exclude the final layer
                x = layer(x)
                if debug and batch_idx == 0 and i < 2:  # Debug first few layers
                    print(f"After FC layer {i} - Features shape: {x.shape}")
            
            if debug and batch_idx == 0:  # Debug first batch
                print(f"Final features shape: {x.shape}")
                print(f"Features sample (first 5 dims): {x[0, :5]}")
            
            features.extend(x.cpu().numpy())
            labels.extend(targets.cpu().numpy().flatten())
    
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    if debug:
        print(f"Extracted features shape: {features_array.shape}")
        print(f"Extracted labels shape: {labels_array.shape}")
        print(f"Feature statistics - Mean: {features_array.mean():.4f}, Std: {features_array.std():.4f}")
        print(f"Labels unique values: {np.unique(labels_array)}")
    
    return features_array, labels_array

def compare_models_performance(
    best_cnn_trainer: ModelTrainer,
    train_loader: DataLoader, 
    val_loader: DataLoader,
    test_loader: DataLoader,
    save_dir: str,
    task_type: str,
    class_names: Optional[Dict[int, str]] = None,
    debug: bool = False
) -> Dict[str, Dict[str, float]]:
    """Compare CNN with traditional ML models."""
    
    if debug:
        print(f"Starting model comparison for task type: {task_type}")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load the best model first
    # Check if best model file exists and load it
    possible_paths = [
        Path('./results/best_model.pth'),
        Path('./results/best_model/best_model.pth'),
        Path(f'./{save_dir.replace("model_evaluation", "")}/best_model.pth'),
        Path('./best_model.pth')
    ]
    
    model_loaded = False
    for model_path in possible_paths:
        if model_path.exists():
            if debug:
                print(f"Loading best model from: {model_path}")
            try:
                best_cnn_trainer.load_model(str(model_path))
                model_loaded = True
                break
            except Exception as e:
                if debug:
                    print(f"Failed to load model from {model_path}: {e}")
                continue
    
    if not model_loaded:
        if debug:
            print("Warning: Best model file not found in any expected location, using current model state")
            print("Searched paths:")
            for path in possible_paths:
                print(f"  - {path}")
    
    # Ensure model is in evaluation mode
    best_cnn_trainer.model.eval()
    
    # Extract features from CNN for traditional ML models
    if debug:
        print("Extracting features for traditional ML models...")
    X_train, y_train = extract_features_for_ml(best_cnn_trainer.model, train_loader, best_cnn_trainer.device, debug)
    X_val, y_val = extract_features_for_ml(best_cnn_trainer.model, val_loader, best_cnn_trainer.device, debug)
    X_test, y_test = extract_features_for_ml(best_cnn_trainer.model, test_loader, best_cnn_trainer.device, debug)
    
    # Check if we have valid features
    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("Extracted features are empty. Check your data loaders and model.")
    
    # Check for NaN or infinite values
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        if debug:
            print("Warning: NaN or infinite values found in training features")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if np.isnan(X_test).any() or np.isinf(X_test).any():
        if debug:
            print("Warning: NaN or infinite values found in test features")
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Combine train and validation sets for traditional ML models
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.hstack([y_train, y_val])
    
    if debug:
        print(f"Combined training set shape: {X_train_combined.shape}")
        print(f"Test set shape: {X_test.shape}")
    
    # Get CNN predictions and probabilities
    if debug:
        print("Getting CNN predictions...")
    y_test_true, y_test_pred_cnn, y_test_probs_cnn = best_cnn_trainer.get_predictions_and_probabilities(test_loader, debug)
    
    # Verify CNN predictions
    if len(y_test_true) == 0 or len(y_test_pred_cnn) == 0:
        raise ValueError("CNN predictions are empty. Check your model and data loader.")
    
    # Initialize models based on task type
    if task_type == 'classification':
        models = {
            'Deep Learning (CNN)': None,  # Will be handled separately
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
    else:  # regression
        models = {
            'Deep Learning (CNN)': None,
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVR(kernel='rbf'),
            'Linear Regression': LinearRegression()
        }
    
    # Train and evaluate traditional ML models
    results = {}
    all_predictions = {'Deep Learning (CNN)': y_test_pred_cnn}
    all_probabilities = {'Deep Learning (CNN)': y_test_probs_cnn}
    
    # CNN metrics
    if debug:
        print("Calculating CNN metrics...")
    if task_type == 'classification':
        try:
            cnn_metrics = {
                'accuracy': np.mean(y_test_true == y_test_pred_cnn),
                'precision': precision_score(y_test_true, y_test_pred_cnn, average='weighted', zero_division=0),
                'recall': recall_score(y_test_true, y_test_pred_cnn, average='weighted', zero_division=0),
                'f1': f1_score(y_test_true, y_test_pred_cnn, average='weighted', zero_division=0)
            }
            
            # Add AUPR and AUC for binary classification
            if len(np.unique(y_test_true)) == 2:
                if y_test_probs_cnn.ndim > 1:
                    y_probs_binary = y_test_probs_cnn[:, 1]  # Use positive class probabilities
                else:
                    y_probs_binary = y_test_probs_cnn
                
                # Check if we have valid probabilities
                if not np.isnan(y_probs_binary).any() and not np.isinf(y_probs_binary).any():
                    cnn_metrics['aupr'] = average_precision_score(y_test_true, y_probs_binary)
                    cnn_metrics['auc'] = roc_auc_score(y_test_true, y_probs_binary)
                else:
                    if debug:
                        print("Warning: Invalid probabilities for CNN AUC/AUPR calculation")
                    cnn_metrics['aupr'] = 0.0
                    cnn_metrics['auc'] = 0.0
            
            if debug:
                print(f"CNN metrics: {cnn_metrics}")
        except Exception as e:
            if debug:
                print(f"Error calculating CNN metrics: {e}")
            cnn_metrics = {metric: 0.0 for metric in ['accuracy', 'precision', 'recall', 'f1', 'aupr', 'auc']}
    else:
        try:
            cnn_metrics = {
                'mse': mean_squared_error(y_test_true, y_test_pred_cnn),
                'mae': mean_absolute_error(y_test_true, y_test_pred_cnn),
                'r2': r2_score(y_test_true, y_test_pred_cnn)
            }
            if debug:
                print(f"CNN metrics: {cnn_metrics}")
        except Exception as e:
            if debug:
                print(f"Error calculating CNN metrics: {e}")
            cnn_metrics = {metric: 0.0 for metric in ['mse', 'mae', 'r2']}
    
    results['Deep Learning (CNN)'] = cnn_metrics
    
    # Train and evaluate traditional ML models
    for model_name, model in models.items():
        if model_name == 'Deep Learning (CNN)':
            continue
            
        if debug:
            print(f"Training {model_name}...")
        
        try:
            # Train the model
            model.fit(X_train_combined, y_train_combined)
            
            # Make predictions
            y_pred = model.predict(X_test)
            all_predictions[model_name] = y_pred
            
            if debug:
                print(f"{model_name} predictions shape: {y_pred.shape}")
                print(f"{model_name} predictions unique values: {np.unique(y_pred)}")
            
            # Calculate metrics
            if task_type == 'classification':
                metrics = {
                    'accuracy': np.mean(y_test_true == y_pred),
                    'precision': precision_score(y_test_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test_true, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test_true, y_pred, average='weighted', zero_division=0)
                }
                
                # Get probabilities for ROC/PR curves
                if hasattr(model, 'predict_proba'):
                    y_probs = model.predict_proba(X_test)
                    all_probabilities[model_name] = y_probs
                    
                    # Add AUPR and AUC for binary classification
                    if len(np.unique(y_test_true)) == 2:
                        if y_probs.ndim > 1:
                            y_probs_binary = y_probs[:, 1]
                        else:
                            y_probs_binary = y_probs
                        
                        if not np.isnan(y_probs_binary).any() and not np.isinf(y_probs_binary).any():
                            metrics['aupr'] = average_precision_score(y_test_true, y_probs_binary)
                            metrics['auc'] = roc_auc_score(y_test_true, y_probs_binary)
                        else:
                            metrics['aupr'] = 0.0
                            metrics['auc'] = 0.0
                else:
                    all_probabilities[model_name] = y_pred  # For SVM without probability
                    metrics['aupr'] = 0.0
                    metrics['auc'] = 0.0
            else:
                metrics = {
                    'mse': mean_squared_error(y_test_true, y_pred),
                    'mae': mean_absolute_error(y_test_true, y_pred),
                    'r2': r2_score(y_test_true, y_pred)
                }
                all_probabilities[model_name] = y_pred
            
            if debug:
                print(f"{model_name} metrics: {metrics}")
            results[model_name] = metrics
            
        except Exception as e:
            if debug:
                print(f"Error training/evaluating {model_name}: {e}")
            if task_type == 'classification':
                results[model_name] = {metric: 0.0 for metric in ['accuracy', 'precision', 'recall', 'f1', 'aupr', 'auc']}
            else:
                results[model_name] = {metric: 0.0 for metric in ['mse', 'mae', 'r2']}
    
    # Plot comparison results
    plot_model_comparison(results, all_predictions, all_probabilities, y_test_true, 
                         save_dir, task_type, class_names)
    
    return results

def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_dir: str,
    task_type: str,
    class_names: Optional[Dict[int, str]] = None
):
    """Plot comparison results between different models."""
    
    # Set scientific paper style
    plt.style.use('default')  # Reset to default first
    
    # Configure matplotlib for scientific papers
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'serif'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 8,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Scientific color palette (more professional)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5D737E', '#8B7355']
    
    save_path = Path(save_dir)
    
    if task_type == 'classification':
        # Plot AUPR and AUC curves for binary classification
        if len(np.unique(y_true)) == 2:
            # AUPR curve
            fig, ax = plt.subplots(figsize=(6, 5))
            for i, (model_name, probs) in enumerate(probabilities.items()):
                if probs.ndim > 1:
                    y_probs_binary = probs[:, 1]
                else:
                    y_probs_binary = probs
                
                precision, recall, _ = precision_recall_curve(y_true, y_probs_binary)
                aupr_score = average_precision_score(y_true, y_probs_binary)
                
                ax.plot(recall, precision, 
                       label=f'{model_name} (AUPR = {aupr_score:.3f})',
                       color=colors[i % len(colors)], linewidth=2.0)
            
            ax.set_xlabel('Recall', fontweight='bold')
            ax.set_ylabel('Precision', fontweight='bold')
            ax.set_title('Precision-Recall Curves', fontweight='bold', pad=20)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(save_path / 'aupr_comparison.png', bbox_inches='tight', facecolor='white')
            plt.close()
            
            # AUC curve
            fig, ax = plt.subplots(figsize=(6, 5))
            for i, (model_name, probs) in enumerate(probabilities.items()):
                if probs.ndim > 1:
                    y_probs_binary = probs[:, 1]
                else:
                    y_probs_binary = probs
                
                fpr, tpr, _ = roc_curve(y_true, y_probs_binary)
                auc_score = roc_auc_score(y_true, y_probs_binary)
                
                ax.plot(fpr, tpr, 
                       label=f'{model_name} (AUC = {auc_score:.3f})',
                       color=colors[i % len(colors)], linewidth=2.0)
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlabel('False Positive Rate', fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontweight='bold')
            ax.set_title('ROC Curves', fontweight='bold', pad=20)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(save_path / 'auc_comparison.png', bbox_inches='tight', facecolor='white')
            plt.close()
        
        # Individual metric bar plots
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in metrics_to_plot:
            if metric in list(results.values())[0]:  # Check if metric exists
                fig, ax = plt.subplots(figsize=(4, 3))
                
                model_names = list(results.keys())
                metric_values = [results[model][metric] for model in model_names]
                
                # Create bars with better styling
                bars = ax.bar(range(len(model_names)), metric_values, 
                             color=colors[:len(model_names)], alpha=0.8, 
                             edgecolor='black', linewidth=0.8)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, metric_values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
                
                ax.set_ylabel(f'{metric.capitalize()}', fontweight='bold')
                ax.set_title(f'{metric.capitalize()} Comparison', fontweight='bold', pad=20)
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=15, ha='right')
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                ax.set_ylim(0, max(metric_values) * 1.15)
                
                # Add subtle background
                ax.set_facecolor('#f8f9fa')
                
                plt.tight_layout()
                plt.savefig(save_path / f'{metric}_comparison.png', bbox_inches='tight', facecolor='white')
                plt.close()
    
    else:  # regression
        # Individual metric bar plots for regression
        metrics_to_plot = ['mse', 'mae', 'r2']
        
        for metric in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            
            model_names = list(results.keys())
            metric_values = [results[model][metric] for model in model_names]
            
            # Create bars with better styling
            bars = ax.bar(range(len(model_names)), metric_values, 
                         color=colors[:len(model_names)], alpha=0.8,
                         edgecolor='black', linewidth=0.8)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, metric_values)):
                height = bar.get_height()
                offset = max(metric_values) * 0.02 if max(metric_values) > 0 else 0.01
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{value:.3f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
            
            ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
            ax.set_title(f'{metric.upper()} Comparison', fontweight='bold', pad=20)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=15, ha='right')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            if max(metric_values) > 0:
                ax.set_ylim(0, max(metric_values) * 1.15)
            
            # Add subtle background
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            plt.savefig(save_path / f'{metric}_comparison.png', bbox_inches='tight', facecolor='white')
            plt.close()
    
    # Reset matplotlib style
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Save comparison results as JSON
    with open(save_path / 'model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Model comparison results saved to {save_dir}") 