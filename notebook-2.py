"""
# Notebook 2: Neural Networks for Scientific Discovery
## Advanced Deep Learning Techniques for Scientific Automation

### Learning Objectives
By the end of this notebook, you will be able to:
- Implement automated neural architecture search
- Apply transfer learning to scientific problems
- Optimize hyperparameters automatically
- Build custom neural networks for scientific data
- Evaluate and validate deep learning models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import timm
from torch.nn import functional as F

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

"""
## Section 1: Custom Scientific Dataset Handling
First, we'll create a flexible dataset class for scientific data.
"""

class ScientificDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        Initialize Scientific Dataset
        
        Parameters:
        -----------
        X : numpy array or pandas DataFrame
            Input features
        y : numpy array or pandas Series
            Target values
        transform : callable
            Optional transform to be applied to samples
        """
        if isinstance(X, pd.DataFrame):
            self.X = torch.FloatTensor(X.values)
        else:
            self.X = torch.FloatTensor(X)
            
        if isinstance(y, pd.Series):
            self.y = torch.FloatTensor(y.values)
        else:
            self.y = torch.FloatTensor(y)
            
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx], self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

"""
## Section 2: Automated Neural Architecture Search
We'll implement a framework for automatically discovering optimal neural network architectures.
"""

class DynamicNet(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_rates):
        """
        Dynamic neural network with configurable architecture
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        layer_sizes : list
            List of integers specifying the size of each hidden layer
        dropout_rates : list
            List of dropout rates for each layer
        """
        super(DynamicNet, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for size, dropout_rate in zip(layer_sizes, dropout_rates):
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size
            
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

class ArchitectureSearch:
    def __init__(self, input_size, n_trials=100):
        self.input_size = input_size
        self.n_trials = n_trials
        
    def objective(self, trial):
        # Define hyperparameters to search
        n_layers = trial.suggest_int('n_layers', 1, 5)
        layer_sizes = [
            trial.suggest_int(f'layer_{i}_size', 32, 512)
            for i in range(n_layers)
        ]
        dropout_rates = [
            trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
            for i in range(n_layers)
        ]
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # Create model and training components
        model = DynamicNet(self.input_size, layer_sizes, dropout_rates)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train and evaluate
        model.train()
        for epoch in range(10):  # Quick training for search
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
            
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_outputs = model(self.X_val)
            val_loss = criterion(val_outputs, self.y_val)
            
        return val_loss.item()
    
    def search(self, X_train, X_val, y_train, y_val):
        """
        Perform architecture search
        
        Returns:
        --------
        best_params : dict
            Best hyperparameters found
        """
        self.X_train = torch.FloatTensor(X_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        self.y_val = torch.FloatTensor(y_val).reshape(-1, 1)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        return study.best_params

"""
## Section 3: Transfer Learning for Scientific Applications
We'll implement transfer learning techniques adapted for scientific data.
"""

class ScientificTransferModel(pl.LightningModule):
    def __init__(self, base_model='resnet18', pretrained=True):
        super(ScientificTransferModel, self).__init__()
        
        # Load pretrained model
        self.base_model = timm.create_model(base_model, pretrained=pretrained)
        
        # Modify for scientific data
        if hasattr(self.base_model, 'fc'):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
    def forward(self, x):
        return self.base_model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

"""
## Section 4: Hyperparameter Optimization
Implementation of advanced hyperparameter optimization strategies.
"""

class HyperparameterOptimizer:
    def __init__(self, model_class, param_space, n_trials=100):
        self.model_class = model_class
        self.param_space = param_space
        self.n_trials = n_trials
        
    def objective(self, trial):
        # Create hyperparameter dictionary
        params = {}
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        # Create and train model
        model = self.model_class(**params)
        trainer = pl.Trainer(
            max_epochs=10,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
            logger=False
        )
        
        trainer.fit(model, self.train_loader, self.val_loader)
        
        # Get validation loss
        val_results = trainer.validate(model, self.val_loader)
        return val_results[0]['val_loss']
    
    def optimize(self, train_loader, val_loader):
        """
        Perform hyperparameter optimization
        
        Returns:
        --------
        best_params : dict
            Best hyperparameters found
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        return study.best_params

"""
## Section 5: Model Evaluation and Validation
Advanced techniques for evaluating scientific deep learning models.
"""

class ModelEvaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def evaluate_uncertainty(self, loader, n_samples=100):
        """
        Evaluate prediction uncertainty using Monte Carlo Dropout
        """
        predictions = []
        self.model.train()  # Enable dropout
        
        with torch.no_grad():
            for _ in range(n_samples):
                batch_predictions = []
                for X, _ in loader:
                    X = X.to(self.device)
                    y_pred = self.model(X)
                    batch_predictions.append(y_pred.cpu().numpy())
                predictions.append(np.concatenate(batch_predictions))
                
        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)
        
        return mean_predictions, std_predictions
    
    def analyze_feature_importance(self, loader):
        """
        Analyze feature importance using integrated gradients
        """
        feature_importance = []
        self.model.eval()
        
        for X, _ in loader:
            X = X.to(self.device)
            X.requires_grad = True
            
            output = self.model(X)
            output.backward(torch.ones_like(output))
            
            gradients = X.grad.abs().mean(dim=0)
            feature_importance.append(gradients.cpu().numpy())
            
        return np.mean(feature_importance, axis=0)
    
    def generate_report(self, test_loader):
        """
        Generate comprehensive evaluation report
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                y_pred = self.model(X)
                predictions.extend(y_pred.cpu().numpy())
                actuals.extend(y.numpy())
                
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        r2 = 1 - (np.sum((actuals - predictions) ** 2) / 
                  np.sum((actuals - actuals.mean()) ** 2))
        
        # Generate plots
        plt.figure(figsize=(15, 5))
        
        # Scatter plot
        plt.subplot(131)
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], 
                [actuals.min(), actuals.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        # Residual plot
        plt.subplot(132)
        residuals = predictions - actuals
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        
        # Error distribution
        plt.subplot(133)
        plt.hist(residuals, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        return {
            'mse': mse,
            'r2': r2,
            'residuals': residuals,
            'predictions': predictions,
            'actuals': actuals
        }

"""
## Section 6: Practical Exercises

1. Architecture Search Exercise:
   - Implement architecture search for a specific scientific dataset
   - Compare different search strategies
   - Analyze the impact of search space decisions

2. Transfer Learning Exercise:
   - Apply transfer learning to a new scientific domain
   - Compare performance with and without transfer learning
   - Analyze which features transfer well between domains

3. Advanced Model Evaluation:
   - Implement uncertainty quantification
   - Analyze feature importance
   - Generate comprehensive evaluation reports

## Additional Resources and References

1. Neural Architecture Search:
   - "Neural Architecture Search: A Survey" (https://arxiv.org/abs/1808.05377)
   - "AutoML: A Survey of the State-of-the-Art" (https://arxiv.org/abs/1908.00709)

2. Transfer Learning:
   - "A Survey on Transfer Learning" (https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)
   - "How transferable are features in deep neural networks?" (https://arxiv.org/abs/1411.1792)

3. Hyperparameter Optimization:
   - "Hyperparameter Optimization: A Review of Algorithms and Applications" (https://arxiv.org/abs/2003.05689)
"""

# Example usage
if __name__ == "__main__":
    # Generate sample data
    X = np.random.randn(1000, 20)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(1000) * 0.1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create dataset
    train_dataset = ScientificDataset(X_train, y_train)
    test_dataset = ScientificDataset(X_test, y_test)
    
    # Example of architecture search
    searcher = ArchitectureSearch(input_size=20)
    best_architecture = searcher.search(X_train, X_test, y_train, y_test)
    print("Best architecture:", best_architecture)
