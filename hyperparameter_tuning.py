import itertools
from typing import Dict, List, Any, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import MedicalCNN, ModelTrainer
import numpy as np

class HyperparameterTuner:
    """Universal class for performing grid search over different model types."""
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_type: Literal['classification', 'regression'],
        model_type: Literal['image', 'ECG', 'voice', 'tabular'] = 'image',  # Future: add 'image3d', 'rnn', 'transformer', etc.
        num_classes: int = 2,
        input_length: int = 5000,  # For ECG and other sequence models
        input_dim: int = None,  # For tabular data
        device: torch.device = None,
        save_dir: str = './grid_search_results'
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type
        self.model_type = model_type
        self.num_classes = num_classes
        self.input_length = input_length
        self.input_dim = input_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict[str, Any]] = []
        
    def _create_model(self, params: Dict[str, Any]) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        """Create model, criterion and optimizer with given parameters based on model type."""
        # Create model based on model type
        if self.model_type == 'image':
            model = MedicalCNN(
                task_type=self.task_type,
                num_classes=self.num_classes,
                num_conv_layers=params['num_conv_layers'],
                conv_channels=params['conv_channels'],
                fc_layers=params['fc_layers']
            )
        elif self.model_type == 'ECG':
            from model import ECG1DCNN
            model = ECG1DCNN(
                task_type=self.task_type,
                num_classes=self.num_classes,
                input_length=self.input_length,
                num_conv_layers=params['num_conv_layers'],
                conv_channels=params['conv_channels'],
                fc_layers=params['fc_layers']
            )
        elif self.model_type == 'voice':
            from model import Voice1DCNN
            model = Voice1DCNN(
                task_type=self.task_type,
                num_classes=self.num_classes,
                input_length=self.input_length,
                num_conv_layers=params['num_conv_layers'],
                conv_channels=params['conv_channels'],
                fc_layers=params['fc_layers']
            )
        elif self.model_type == 'tabular':
            from model import ASDTabularModel
            model = ASDTabularModel(
                task_type=self.task_type,
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                hidden_layers=params['hidden_layers'],
                dropout_rate=params.get('dropout_rate', 0.5)
            )
        # TODO: Add support for future model types here
        # elif self.model_type == 'image3d':
        #     model = Medical3DCNN(...)
        # elif self.model_type == 'rnn':
        #     model = MedicalRNN(...)
        # elif self.model_type == 'transformer':
        #     model = MedicalTransformer(...)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: 'image', 'ECG', 'voice', 'tabular'")
        
        # Define loss function based on task type
        if self.task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate']
        )
            
        return model, criterion, optimizer
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]] = None,
        num_epochs: int = 50,
        early_stopping_patience: int = 15
    ) -> Dict[str, Any]:
        """Perform grid search over hyperparameters."""
        if param_grid is None:
            if self.model_type == 'tabular':
                param_grid = {
                    'hidden_layers': [[512, 128], [256, 128, 64], [1024, 512, 256], [128, 64]],
                    'dropout_rate': [0.3, 0.5, 0.7],
                    'learning_rate': [0.001, 0.0001, 0.01]
                }
            else:
                param_grid = {
                    'num_conv_layers': [3, 4, 5],
                    'conv_channels': [32, 64, 128],
                    'fc_layers': [[512, 128], [256, 64], [1024, 256, 64]],
                    'learning_rate': [0.001, 0.0001]
                }
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        for i, params in enumerate(param_combinations):
            print(f"\nTrying combination {i+1}/{len(param_combinations)}:")
            print(json.dumps(params, indent=2))
            
            # Create model and trainer
            model, criterion, optimizer = self._create_model(params)
            trainer = ModelTrainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                task_type=self.task_type
            )
            
            # Train model
            history = trainer.train(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=num_epochs,
                save_dir=str(self.save_dir / f"combination_{i+1}"),
                early_stopping_patience=early_stopping_patience
            )
            trainer.plot_training_history(str(self.save_dir / f"combination_{i+1}"))
            
            # Get best validation metrics
            best_val_loss = min(trainer.val_losses)
            best_val_metrics = trainer.val_metrics[np.argmin(trainer.val_losses)]
            
            # Store results
            result = {
                'params': params,
                'best_val_loss': best_val_loss,
                'best_val_metrics': best_val_metrics,
                'history': history
            }
            self.results.append(result)
            
            # Save result to file
            with open(self.save_dir / f'combination_{i+1}_results.json', 'w') as f:
                json.dump(result, f, indent=4)
        
        # Find best combination
        best_idx = np.argmin([r['best_val_loss'] for r in self.results])
        best_result = self.results[best_idx]
        
        # Save summary
        summary = {
            'best_combination': best_result['params'],
            'best_val_loss': best_result['best_val_loss'],
            'best_val_metrics': best_result['best_val_metrics'],
            'all_results': self.results
        }
        
        with open(self.save_dir / 'grid_search_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        return summary
    
    def plot_results(self, metric: str = 'best_val_loss') -> None:
        """Plot grid search results using boxplots to show distribution of metrics across different parameter values."""
        df = self.get_results_dataframe()
        df.to_csv(self.save_dir / 'grid_search_results.csv', index=False)   
        
        # Select numerical columns for plotting
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols if col != metric]
        
        # Create subplots for each numerical parameter
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        if n_cols == 1:
            axes = [axes]
        
        for ax, param in zip(axes, numerical_cols):
            # Convert parameter values to categorical for better boxplot visualization
            df[param] = df[param].astype(str)
            sns.boxplot(data=df, x=param, y=metric, ax=ax)
            # ax.set_title(f'Distribution of {metric} by {param}')
            ax.set_xlabel(param)
            ax.set_ylabel(metric)
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'grid_search_results.png')
        plt.close()
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        # Flatten the results for DataFrame creation
        flattened_results = []
        for result in self.results:
            flat_result = {
                **result['params'],
                'best_val_loss': result['best_val_loss'],
                **{f'best_val_{k}': v for k, v in result['best_val_metrics'].items()}
            }
            flattened_results.append(flat_result)
        
        return pd.DataFrame(flattened_results)
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get the best model parameters based on validation loss."""
        if not self.results:
            raise ValueError("No results available. Run grid_search first.")
        
        best_result = min(self.results, key=lambda x: x['best_val_loss'])
        return best_result

 