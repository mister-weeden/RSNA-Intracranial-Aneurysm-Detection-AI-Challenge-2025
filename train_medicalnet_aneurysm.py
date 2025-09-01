#!/usr/bin/env python3
"""
MedicalNet Training Script for Aneurysm Detection
Binary classification: aneurysm present vs. absent
Tracks AUC, sensitivity, specificity, FPR, TNR
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

# Add MedicalNet to path
sys.path.append(str(Path(__file__).parent / 'models' / 'MedicalNet'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AneurysmDataset(Dataset):
    """Dataset for aneurysm detection (binary classification)"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load file lists
        self.data_files = list(self.data_dir.glob('*.nii*'))
        
        # For now, simulate labels (in real case, load from annotations)
        # Files with certain patterns might be positive cases
        self.labels = []
        for f in self.data_files:
            # Simulate: files with 'cowseg' or certain IDs are positive
            if 'cowseg' in f.name or random.random() < 0.3:  # 30% positive rate
                self.labels.append(1)  # Aneurysm present
            else:
                self.labels.append(0)  # No aneurysm
        
        logger.info(f"{split} set: {len(self.data_files)} files, "
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