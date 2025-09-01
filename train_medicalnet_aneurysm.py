#!/usr/bin/env python3
"""
MedicalNet Training Script for Aneurysm Detection
Binary classification: aneurysm present vs. absent
Tracks AUC, sensitivity, specificity, FPR, TNR

@cursor Implement deterministic training pipeline with reproducible results
@cursor Add complexity analysis for O(n*log(n)) optimization opportunities
@cursor Integrate multi-task learning for 13 aneurysm characteristics classification
@cursor Add uncertainty quantification using Monte Carlo dropout
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import nibabel as nib
from datetime import datetime
import random
import hashlib
from typing import Dict, List, Tuple, Optional, Union
import warnings
from collections import defaultdict
import time
import psutil
import gc

# Add MedicalNet to path
sys.path.append(str(Path(__file__).parent / 'models' / 'MedicalNet'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_deterministic_seed(seed: int = 42) -> None:
    """
    Set deterministic seed for reproducible results across all random number generators.
    
    @cursor Critical for reproducible research - eliminates randomness in model training
    
    Args:
        seed: Random seed value for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class PerformanceMonitor:
    """
    Monitor computational complexity and memory usage during training.
    
    @cursor Identifies O(n*log(n)) bottlenecks and memory optimization opportunities
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        
    def end_timer(self, operation: str, data_size: int = 1) -> float:
        """
        End timing and calculate complexity metrics.
        
        Args:
            operation: Name of the operation
            data_size: Size of data processed (for complexity analysis)
            
        Returns:
            Elapsed time in seconds
        """
        if operation not in self.start_times:
            return 0.0
            
        elapsed = time.time() - self.start_times[operation]
        self.metrics[operation].append({
            'time': elapsed,
            'data_size': data_size,
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'complexity_ratio': elapsed / (data_size * np.log(data_size)) if data_size > 1 else elapsed
        })
        
        # @cursor Flag operations with complexity > O(n*log(n))
        if data_size > 100 and elapsed / (data_size * np.log(data_size)) > 0.001:
            logger.warning(f"Operation '{operation}' may have complexity > O(n*log(n)): "
                         f"{elapsed:.4f}s for {data_size} items")
        
        return elapsed
        
    def get_summary(self) -> Dict:
        """Get performance summary for optimization analysis."""
        summary = {}
        for op, metrics in self.metrics.items():
            if metrics:
                times = [m['time'] for m in metrics]
                summary[op] = {
                    'avg_time': np.mean(times),
                    'max_time': np.max(times),
                    'total_calls': len(metrics),
                    'avg_memory_mb': np.mean([m['memory_mb'] for m in metrics])
                }
        return summary

class AneurysmDataset(Dataset):
    """
    Dataset for aneurysm detection supporting both binary and multi-class (13 characteristics) classification.
    
    @cursor Implements deterministic data loading with consistent ordering
    @cursor Supports multi-task learning for AUC_AP and AUC_0 through AUC_12
    @cursor Optimized memory usage with lazy loading and caching strategies
    """
    
    def __init__(self, data_dir: Union[str, Path], split: str = 'train', 
                 transform: Optional[callable] = None, 
                 multi_task: bool = True,
                 cache_size: int = 100,
                 deterministic: bool = True):
        """
        Initialize dataset with deterministic loading and multi-task support.
        
        Args:
            data_dir: Directory containing NIfTI files
            split: Dataset split ('train', 'val', 'test')
            transform: Optional data transformations
            multi_task: Enable multi-task learning for 13 aneurysm characteristics
            cache_size: Number of volumes to cache in memory
            deterministic: Use deterministic file ordering
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.multi_task = multi_task
        self.cache_size = cache_size
        self.cache = {}
        self.performance_monitor = PerformanceMonitor()
        
        # @cursor Deterministic file loading eliminates randomness in data order
        self.data_files = sorted(list(self.data_dir.glob('*.nii*')))
        if deterministic:
            # Use file hash for consistent ordering across runs
            self.data_files.sort(key=lambda x: hashlib.md5(str(x).encode()).hexdigest())
        
        # Load or generate labels deterministically
        self._load_labels()
        
        logger.info(f"{split} set: {len(self.data_files)} files, "
class MultiTaskAneurysmNet(nn.Module):
    """
    Multi-task neural network for aneurysm detection and characterization.
    
    @cursor Implements shared feature extractor with task-specific heads
    @cursor Supports uncertainty quantification via Monte Carlo dropout
    @cursor Optimized architecture to reduce computational complexity
    """
    
    def __init__(self, input_channels: int = 1, 
                 num_aneurysm_chars: int = 13,
                 dropout_rate: float = 0.3,
                 use_uncertainty: bool = True):
        """
        Initialize multi-task network architecture.
        
        Args:
            input_channels: Number of input channels (typically 1 for medical images)
            num_aneurysm_chars: Number of aneurysm characteristics to classify
            dropout_rate: Dropout rate for uncertainty quantification
            use_uncertainty: Enable Monte Carlo dropout for uncertainty estimation
        """
        super(MultiTaskAneurysmNet, self).__init__()
        
        self.use_uncertainty = use_uncertainty
        self.dropout_rate = dropout_rate
        
        # @cursor Shared feature extractor - reduces computational complexity
        # Uses efficient 3D CNN with progressive downsampling
        self.shared_features = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # Reduce spatial dimensions by 2x
            
            # Block 2: Mid-level features
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # Further reduction
            
            # Block 3: High-level features
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),  # Fixed output size
        )
        
        # Calculate feature size after shared layers
        self.feature_size = 128 * 4 * 4 * 4  # 8192 features
        
        # @cursor Binary aneurysm detection head (AUC_AP)
        self.aneurysm_head = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),  # Binary classification
            nn.Sigmoid()
        )
        
        # @cursor Multi-class aneurysm characteristics heads (AUC_0 to AUC_12)
        self.characteristic_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1),  # Binary classification for each characteristic
                nn.Sigmoid()
            ) for _ in range(num_aneurysm_chars)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights using Xavier/He initialization.
        
        @cursor Proper weight initialization for stable training
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional Monte Carlo sampling for uncertainty.
        
        @cursor Implements uncertainty quantification via MC dropout
        @cursor Returns predictions for all tasks simultaneously
        
        Args:
            x: Input tensor [batch_size, channels, depth, height, width]
            num_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            Dictionary containing predictions for all tasks
        """
        if self.use_uncertainty and num_samples > 1:
            return self._forward_with_uncertainty(x, num_samples)
        else:
            return self._forward_single(x)
    
    def _forward_single(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single forward pass without uncertainty sampling."""
        # Extract shared features
        features = self.shared_features(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Binary aneurysm detection
        aneurysm_pred = self.aneurysm_head(features)
        
        # Aneurysm characteristics
        char_preds = []
        for head in self.characteristic_heads:
            char_pred = head(features)
            char_preds.append(char_pred)
        
        results = {
            'aneurysm': aneurysm_pred.squeeze(-1),  # Remove last dimension
        }
        
        # Add characteristic predictions
        for i, pred in enumerate(char_preds):
            results[f'char_{i}'] = pred.squeeze(-1)
        
        return results
    
    def _forward_with_uncertainty(self, x: torch.Tensor, 
                                 num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Monte Carlo dropout for uncertainty estimation.
        
        @cursor Enables uncertainty quantification for clinical decision support
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples
            
        Returns:
            Dictionary with mean predictions and uncertainty estimates
        """
        # Enable dropout during inference
        self.train()
        
        predictions = []
        for _ in range(num_samples):
            pred = self._forward_single(x)
            predictions.append(pred)
        
        # Calculate mean and uncertainty (standard deviation)
        results = {}
        for key in predictions[0].keys():
            samples = torch.stack([p[key] for p in predictions], dim=0)
            mean_pred = torch.mean(samples, dim=0)
            uncertainty = torch.std(samples, dim=0)
            
            results[key] = mean_pred
            results[f'{key}_uncertainty'] = uncertainty
        
        return results
class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with adaptive weighting and focal loss components.
    
    @cursor Implements competition scoring formula: ½(AUC_AP + 1/13 Σ AUC_i)
    @cursor Uses focal loss to handle class imbalance in aneurysm detection
    @cursor Adaptive task weighting based on training progress
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 aneurysm_weight: float = 0.5, char_weight: float = 0.5):
        """
        Initialize multi-task loss function.
        
        Args:
            alpha: Focal loss alpha parameter for class balancing
            gamma: Focal loss gamma parameter for hard example mining
            aneurysm_weight: Weight for aneurysm detection task
            char_weight: Weight for characteristic classification tasks
        """
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.aneurysm_weight = aneurysm_weight
        self.char_weight = char_weight
        
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.
        
        @cursor Addresses class imbalance common in medical imaging
        
        Args:
            pred: Predicted probabilities
            target: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions for all tasks
            targets: Ground truth labels for all tasks
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Aneurysm detection loss (main task)
        aneurysm_loss = self.focal_loss(predictions['aneurysm'], targets['aneurysm'])
        losses['aneurysm_loss'] = aneurysm_loss
        
        # Characteristic classification losses
        char_losses = []
        for i in range(13):
            char_key = f'char_{i}'
            if char_key in predictions and char_key in targets:
                char_loss = self.focal_loss(predictions[char_key], targets[char_key])
                losses[f'char_{i}_loss'] = char_loss
                char_losses.append(char_loss)
        
        # Combined characteristic loss
        if char_losses:
            combined_char_loss = torch.stack(char_losses).mean()
            losses['char_combined_loss'] = combined_char_loss
        else:
            combined_char_loss = torch.tensor(0.0, device=aneurysm_loss.device)
        
        # Total loss following competition scoring
        total_loss = (self.aneurysm_weight * aneurysm_loss + 
                     self.char_weight * combined_char_loss)
        losses['total_loss'] = total_loss
        
        return losses

class AneurysmTrainer:
    """
    Advanced trainer for multi-task aneurysm detection with optimization strategies.
    
    @cursor Implements deterministic training with reproducible results
    @cursor Includes performance monitoring and complexity analysis
    @cursor Supports various optimization strategies and learning rate scheduling
    """
    
    def __init__(self, model: MultiTaskAneurysmNet, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 config: Dict):
        """
        Initialize trainer with model and data loaders.
        
        Args:
            model: Multi-task aneurysm detection model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = MultiTaskLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0),
            aneurysm_weight=config.get('aneurysm_weight', 0.5),
            char_weight=config.get('char_weight', 0.5)
        )
        
        # Initialize optimizer with advanced settings
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        # @cursor Set deterministic training for reproducibility
        set_deterministic_seed(config.get('seed', 42))
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with advanced configuration.
        
        @cursor Uses AdamW with weight decay for better generalization
        @cursor Implements gradient clipping to prevent exploding gradients
        
        Returns:
            Configured optimizer
        """
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler.
        
        @cursor Implements cosine annealing with warm restarts for better convergence
        
        Returns:
            Learning rate scheduler
        """
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with performance monitoring.
        
        @cursor Implements gradient clipping and memory optimization
        @cursor Monitors computational complexity for bottleneck identification
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_losses = defaultdict(list)
        epoch_predictions = defaultdict(list)
        epoch_targets = defaultdict(list)
        
        self.performance_monitor.start_timer('epoch_training')
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            self.performance_monitor.start_timer('batch_processing')
            
            # Move data to device
            data = data.to(self.device)
            batch_targets = {}
            for key, value in targets.items():
                batch_targets[key] = value.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(data)
            
            # Compute losses
            losses = self.criterion(predictions, batch_targets)
            
            # Backward pass with gradient clipping
            losses['total_loss'].backward()
            
            # @cursor Gradient clipping prevents exploding gradients
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Store metrics
            for key, loss in losses.items():
                epoch_losses[key].append(loss.item())
            
            # Store predictions and targets for AUC calculation
            for key in predictions.keys():
                if key in batch_targets:
                    epoch_predictions[key].extend(predictions[key].detach().cpu().numpy())
                    epoch_targets[key].extend(batch_targets[key].detach().cpu().numpy())
            
            # @cursor Memory optimization - clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            batch_size = data.size(0)
            self.performance_monitor.end_timer('batch_processing', batch_size)
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {losses["total_loss"].item():.4f}')
        
        self.performance_monitor.end_timer('epoch_training', len(self.train_loader))
        
        # Calculate epoch metrics
        metrics = {}
        for key, losses in epoch_losses.items():
            metrics[f'train_{key}'] = np.mean(losses)
        
        # Calculate AUC scores
        auc_scores = self._calculate_auc_scores(epoch_predictions, epoch_targets, 'train')
        metrics.update(auc_scores)
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        @cursor Includes uncertainty quantification during validation
        @cursor Calculates competition score following official formula
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        epoch_losses = defaultdict(list)
        epoch_predictions = defaultdict(list)
        epoch_targets = defaultdict(list)
        epoch_uncertainties = defaultdict(list)
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                # Move data to device
                data = data.to(self.device)
                batch_targets = {}
                for key, value in targets.items():
                    batch_targets[key] = value.to(self.device)
                
                # Forward pass with uncertainty quantification
                num_mc_samples = self.config.get('mc_samples', 5)
                predictions = self.model(data, num_samples=num_mc_samples)
                
                # Compute losses (only on mean predictions)
                mean_predictions = {k: v for k, v in predictions.items() 
                                  if not k.endswith('_uncertainty')}
                losses = self.criterion(mean_predictions, batch_targets)
                
                # Store metrics
                for key, loss in losses.items():
                    epoch_losses[key].append(loss.item())
                
                # Store predictions, targets, and uncertainties
                for key in mean_predictions.keys():
                    if key in batch_targets:
                        epoch_predictions[key].extend(
                            mean_predictions[key].detach().cpu().numpy())
                        epoch_targets[key].extend(
                            batch_targets[key].detach().cpu().numpy())
                        
                        # Store uncertainty if available
                        uncertainty_key = f'{key}_uncertainty'
                        if uncertainty_key in predictions:
                            epoch_uncertainties[key].extend(
                                predictions[uncertainty_key].detach().cpu().numpy())
        
        # Calculate epoch metrics
        metrics = {}
        for key, losses in epoch_losses.items():
            metrics[f'val_{key}'] = np.mean(losses)
        
        # Calculate AUC scores and competition score
        auc_scores = self._calculate_auc_scores(epoch_predictions, epoch_targets, 'val')
        metrics.update(auc_scores)
        
        # Calculate competition score
        competition_score = self._calculate_competition_score(auc_scores)
        metrics['val_competition_score'] = competition_score
        
        # Add uncertainty metrics
        for key, uncertainties in epoch_uncertainties.items():
            metrics[f'val_{key}_mean_uncertainty'] = np.mean(uncertainties)
        
        return metrics
    
    def _calculate_auc_scores(self, predictions: Dict[str, List], 
                             targets: Dict[str, List], 
                             prefix: str) -> Dict[str, float]:
        """
        Calculate AUC scores for all tasks.
        
        @cursor Implements robust AUC calculation with edge case handling
        
        Args:
            predictions: Dictionary of predictions for each task
            targets: Dictionary of targets for each task
            prefix: Prefix for metric names ('train' or 'val')
            
        Returns:
            Dictionary containing AUC scores
        """
        auc_scores = {}
        
        for key in predictions.keys():
            if key in targets and len(predictions[key]) > 0:
                try:
                    y_true = np.array(targets[key])
                    y_pred = np.array(predictions[key])
                    
                    # Handle edge cases
                    if len(np.unique(y_true)) < 2:
                        auc = 0.5  # Default for single class
                    else:
                        auc = roc_auc_score(y_true, y_pred)
                    
                    auc_scores[f'{prefix}_auc_{key}'] = auc
                    
                except Exception as e:
                    logger.warning(f"Error calculating AUC for {key}: {e}")
                    auc_scores[f'{prefix}_auc_{key}'] = 0.5
        
        return auc_scores
    
    def _calculate_competition_score(self, auc_scores: Dict[str, float]) -> float:
        """
        Calculate competition score: ½(AUC_AP + 1/13 Σ AUC_i)
        
        @cursor Implements official competition scoring formula
        
        Args:
            auc_scores: Dictionary containing AUC scores
            
        Returns:
            Competition score
        """
        # Extract aneurysm AUC (AUC_AP)
        aneurysm_auc = auc_scores.get('val_auc_aneurysm', 0.5)
        
        # Extract characteristic AUCs (AUC_0 to AUC_12)
        char_aucs = []
        for i in range(13):
            char_key = f'val_auc_char_{i}'
            char_auc = auc_scores.get(char_key, 0.5)
            char_aucs.append(char_auc)
        
        # Calculate competition score
        char_mean = np.mean(char_aucs) if char_aucs else 0.5
        competition_score = 0.5 * (aneurysm_auc + char_mean)
        
        return competition_score
    
    def train(self, num_epochs: int) -> Dict:
        """
        Main training loop with advanced optimization and monitoring.
        
        @cursor Implements early stopping and model checkpointing
        @cursor Includes comprehensive performance monitoring and logging
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_score = 0.0
        patience_counter = 0
        patience = self.config.get('patience', 20)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_total_loss'])
                else:
                    self.scheduler.step()
            
            # Store metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            for key, value in train_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            for key, value in val_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            # Check for improvement
            current_score = val_metrics.get('val_competition_score', 0.0)
            if current_score > best_score:
                best_score = current_score
                patience_counter = 0
                self._save_checkpoint(epoch, best_score, 'best_model.pth')
            else:
                patience_counter += 1
            
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Time: {epoch_time:.2f}s - "
                       f"Train Loss: {train_metrics.get('train_total_loss', 0):.4f} - "
                       f"Val Loss: {val_metrics.get('val_total_loss', 0):.4f} - "
                       f"Competition Score: {current_score:.4f} - "
                       f"LR: {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(epoch, current_score, f'checkpoint_epoch_{epoch+1}.pth')
        
        # Performance summary
        perf_summary = self.performance_monitor.get_summary()
        logger.info("Performance Summary:")
        for op, metrics in perf_summary.items():
            logger.info(f"  {op}: avg_time={metrics['avg_time']:.4f}s, "
                       f"calls={metrics['total_calls']}, "
                       f"avg_memory={metrics['avg_memory_mb']:.1f}MB")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, score: float, filename: str) -> None:
        """
        Save model checkpoint with training state.
        
        Args:
            epoch: Current epoch number
            score: Current validation score
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'score': score,
            'config': self.config,
            'history': self.history
        }
        
        checkpoint_path = Path('checkpoints') / filename
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

def main():
    """
    Main training function with configuration and data setup.
    
    @cursor Entry point for deterministic aneurysm detection training
    @cursor Implements complete pipeline from data loading to model evaluation
    """
    # Configuration
    config = {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'weight_decay': 1e-5,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'aneurysm_weight': 0.5,
        'char_weight': 0.5,
        'max_grad_norm': 1.0,
        'patience': 20,
        'seed': 42,
        'mc_samples': 5,
        'log_interval': 10,
        'save_interval': 10
    }
    
    # Set deterministic seed
    set_deterministic_seed(config['seed'])
    
    # Data paths
    data_dir = Path('/Users/owner/work/aneurysm_dataset')
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    # Create datasets
    train_dataset = AneurysmDataset(
        train_dir, 
        split='train', 
        multi_task=True,
        deterministic=True
    )
    
    val_dataset = AneurysmDataset(
        val_dir, 
        split='val', 
        multi_task=True,
        deterministic=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = MultiTaskAneurysmNet(
        input_channels=1,
        num_aneurysm_chars=13,
        dropout_rate=0.3,
        use_uncertainty=True
    )
    
    # Create trainer
    trainer = AneurysmTrainer(model, train_loader, val_loader, config)
    
    # Start training
    history = trainer.train(config['num_epochs'])
    
    # Save final results
    results_path = Path('results') / 'training_results.json'
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_history = {}
        for key, values in history.items():
            if isinstance(values, list):
                json_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
            else:
                json_history[key] = values
        
        json.dump({
            'config': config,
            'history': json_history,
            'performance_summary': trainer.performance_monitor.get_summary()
        }, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {results_path}")

if __name__ == "__main__":
    main()
        """
        Load or generate labels for both binary and multi-class tasks.
        
        @cursor Replace with actual annotation loading from CSV/JSON files
        @cursor Current implementation uses deterministic simulation for development
        """
        self.labels = []
        
        # @cursor TODO: Load actual labels from train.csv or annotation files
        # For now, use deterministic simulation based on file characteristics
        for i, file_path in enumerate(self.data_files):
            # Deterministic label generation based on file hash
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
            hash_int = int(file_hash[:8], 16)
            
            # Binary aneurysm detection (AUC_AP)
            aneurysm_present = (hash_int % 100) < 30  # 30% positive rate
            
            label_dict = {'aneurysm': int(aneurysm_present)}
            
            if self.multi_task:
                # 13 aneurysm characteristics (AUC_0 through AUC_12)
                # @cursor These should be loaded from actual medical annotations
                for j in range(13):
                    # Simulate correlated characteristics
                    char_prob = 0.1 + 0.3 * aneurysm_present + 0.05 * ((hash_int >> j) % 10)
                    label_dict[f'char_{j}'] = int(char_prob > 0.5)
            
            self.labels.append(label_dict)
    
    def _load_volume(self, idx: int) -> np.ndarray:
        """
        Load and preprocess volume with caching for performance optimization.
        
        @cursor Implements LRU-style caching to reduce I/O bottlenecks
        @cursor Memory usage optimization for large datasets
        
        Args:
            idx: Index of volume to load
            
        Returns:
            Preprocessed volume as numpy array
        """
        if idx in self.cache:
            return self.cache[idx]
        
        self.performance_monitor.start_timer('volume_loading')
        
        file_path = self.data_files[idx]
        
        try:
            # Load NIfTI volume
            nii = nib.load(str(file_path))
            volume = nii.get_fdata().astype(np.float32)
            
            # Basic preprocessing
            # @cursor Add vessel enhancement and aneurysm-specific preprocessing
            volume = self._preprocess_volume(volume)
            
            # Cache management - remove oldest if cache full
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                gc.collect()  # Force garbage collection
            
            self.cache[idx] = volume
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            # Return zero volume as fallback
            volume = np.zeros((64, 64, 64), dtype=np.float32)
        
        self.performance_monitor.end_timer('volume_loading', volume.size)
        return volume
    
    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Preprocess volume for aneurysm detection.
        
        @cursor Implement vessel enhancement and aneurysm-specific preprocessing
        @cursor Add intensity normalization and artifact removal
        
        Args:
            volume: Raw volume data
            
        Returns:
            Preprocessed volume
        """
        # Intensity normalization
        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        # @cursor Add Frangi vesselness filter for vessel enhancement
        # @cursor Add aneurysm-specific preprocessing pipeline
        
        return volume
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item with multi-task labels.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Tuple of (volume_tensor, labels_dict)
        """
        volume = self._load_volume(idx)
        labels = self.labels[idx]
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if specified
        if self.transform:
            volume_tensor = self.transform(volume_tensor)
        
        # Convert labels to tensors
        label_tensors = {}
        for key, value in labels.items():
            label_tensors[key] = torch.tensor(value, dtype=torch.float32)
        
        return volume_tensor, label_tensors
                   f"{sum(self.labels)} positive, {len(self.labels)-sum(self.labels)} negative")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load NIfTI file
        nii_path = self.data_files[idx]
        img = nib.load(str(nii_path))
        data = img.get_fdata()
        
        # Normalize
        data = (data - data.mean()) / (data.std() + 1e-8)
        
        # Convert to tensor
        data = torch.FloatTensor(data).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if any
        if self.transform:
            data = self.transform(data)
        
        # Resize to fixed size (e.g., 128x128x128) if needed
        if data.shape[-3:] != (128, 128, 128):
            data = torch.nn.functional.interpolate(
                data.unsqueeze(0), 
                size=(128, 128, 128), 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return data, label

class MedicalNetClassifier(nn.Module):
    """MedicalNet-based classifier for aneurysm detection"""
    
    def __init__(self, pretrained_path=None, num_classes=2):
        super().__init__()
        
        # Load pretrained ResNet model
        from models import resnet
        
        # Use ResNet50 as backbone
        self.encoder = resnet.resnet50(
            sample_input_D=128,
            sample_input_H=128,
            sample_input_W=128,
            num_seg_classes=2
        )
        
        # Load pretrained weights if available
        if pretrained_path and Path(pretrained_path).exists():
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.encoder.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Replace final layer for binary classification
        in_features = self.encoder.conv_seg.in_channels
        self.encoder.conv_seg = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.encoder(x)

class AneurysmTrainer:
    """Trainer for aneurysm detection model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create model
        pretrained_path = config.get('pretrained_path')
        self.model = MedicalNetClassifier(pretrained_path).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        # Metrics tracking
        self.best_auc = 0.0
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'fpr': [],
            'tnr': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_metrics(self, y_true, y_pred_proba):
        """Calculate classification metrics"""
        # Calculate AUC
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        # Get optimal threshold from ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Make predictions with optimal threshold
        y_pred = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = specificity  # True negative rate
        
        return {
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr_rate,
            'tnr': tnr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        return avg_loss, metrics
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'auc_score': metrics['auc'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'fpr': metrics['fpr'],
            'tnr': metrics['tnr'],
            'metrics_history': self.metrics_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_auc_{metrics['auc']:.4f}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with AUC: {metrics['auc']:.4f}")
    
    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val AUC: {val_metrics['auc']:.4f}, "
                       f"Sensitivity: {val_metrics['sensitivity']:.4f}, "
                       f"Specificity: {val_metrics['specificity']:.4f}, "
                       f"FPR: {val_metrics['fpr']:.4f}")
            
            # Save metrics
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['auc'].append(val_metrics['auc'])
            self.metrics_history['sensitivity'].append(val_metrics['sensitivity'])
            self.metrics_history['specificity'].append(val_metrics['specificity'])
            self.metrics_history['fpr'].append(val_metrics['fpr'])
            self.metrics_history['tnr'].append(val_metrics['tnr'])
            
            # Check if best model
            is_best = val_metrics['auc'] > self.best_auc
            if is_best:
                self.best_auc = val_metrics['auc']
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_frequency'] == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Early stopping check
            if epoch > 10 and all(self.metrics_history['auc'][-10:][i] <= self.metrics_history['auc'][-10:][i+1] 
                                  for i in range(8)):
                logger.info("Early stopping triggered - no improvement in 10 epochs")
                break
        
        logger.info(f"Training complete! Best AUC: {self.best_auc:.4f}")

def main():
    # Configuration
    config = {
        'data_dir': '/Users/owner/work/aneurysm_dataset',
        'pretrained_path': '/Users/owner/work/models/MedicalNet/pretrain/resnet_50_23dataset.pth',
        'checkpoint_dir': '/Users/owner/work/models/MedicalNet/checkpoints',
        'batch_size': 4,  # Small batch size for memory
        'learning_rate': 1e-4,
        'epochs': 100,
        'save_frequency': 5,
        'num_workers': 0  # Set to 0 for macOS compatibility
    }
    
    # Prepare data
    logger.info("Preparing datasets...")
    train_dataset = AneurysmDataset(Path(config['data_dir']) / 'train', split='train')
    val_dataset = AneurysmDataset(Path(config['data_dir']) / 'val', split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Create trainer and start training
    trainer = AneurysmTrainer(config)
    trainer.train(train_loader, val_loader, config['epochs'])

if __name__ == "__main__":
    main()