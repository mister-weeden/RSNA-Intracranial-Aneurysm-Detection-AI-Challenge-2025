#!/usr/bin/env python3
"""
3D U-Net Training Script for RSNA Aneurysm Detection
Handles NIfTI volumes with vessel segmentation masks
"""

import os
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class AneurysmDataset(Dataset):
    """Dataset for loading NIfTI volumes and segmentation masks"""
    
    def __init__(self, csv_path, data_dir, transform=None, use_segmentation=True):
        self.df = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_segmentation = use_segmentation
        
        # Target columns for multi-label classification
        self.target_columns = [
            'Left Infraclinoid Internal Carotid Artery',
            'Right Infraclinoid Internal Carotid Artery',
            'Left Supraclinoid Internal Carotid Artery',
            'Right Supraclinoid Internal Carotid Artery',
            'Left Middle Cerebral Artery',
            'Right Middle Cerebral Artery',
            'Anterior Communicating Artery',
            'Left Anterior Cerebral Artery',
            'Right Anterior Cerebral Artery',
            'Left Posterior Communicating Artery',
            'Right Posterior Communicating Artery',
            'Basilar Tip',
            'Other Posterior Circulation',
            'Aneurysm Present'
        ]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row['SeriesInstanceUID']
        
        # Load NIfTI volume
        volume_path = self.data_dir / f"{series_id}.nii.gz"
        volume = nib.load(str(volume_path)).get_fdata()
        
        # Normalize volume
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        # Load segmentation mask if available
        if self.use_segmentation:
            seg_path = self.data_dir / f"{series_id}_cowseg.nii.gz"
            if seg_path.exists():
                segmentation = nib.load(str(seg_path)).get_fdata()
                # Stack volume and segmentation as 2 channels
                data = np.stack([volume, segmentation], axis=0)
            else:
                # If no segmentation, duplicate volume
                data = np.stack([volume, volume], axis=0)
        else:
            data = np.expand_dims(volume, 0)
        
        # Get labels
        labels = torch.tensor(
            row[self.target_columns].values.astype(np.float32)
        )
        
        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)
        
        # Resize to standard shape if needed (e.g., 256x256x128)
        # This is a simplified example - you'd want proper resampling
        if data.shape != (2, 256, 256, 128):
            # Implement proper resampling here
            pass
        
        return data, labels, series_id

class UNet3D(nn.Module):
    """Simple 3D U-Net implementation"""
    
    def __init__(self, in_channels=2, out_channels=14, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature * 2, feature))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(features[0], 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_channels),
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for i, encode in enumerate(self.encoder):
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](x)
        
        # Classification
        return self.classifier(x)

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, labels, _) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, labels, _ in tqdm(dataloader, desc="Validating"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main(args):
    # Setup
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting training with args: {args}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create data loaders
    train_dataset = AneurysmDataset(
        args.train_csv,
        args.data_dir,
        use_segmentation=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Split for validation (or use separate validation CSV)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = UNet3D(in_channels=2, out_channels=14).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if args.mixed_precision else None
    
    # Training loop
    best_val_loss = float('inf')
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, logger
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = f"{args.model_save_dir}/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"{args.model_save_dir}/best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D U-Net for Aneurysm Detection")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to NIfTI files")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--localizers_csv", type=str, help="Path to localizers CSV")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_save_dir", type=str, default="models/", help="Model save directory")
    parser.add_argument("--log_dir", type=str, default="logs/", help="Log directory")
    parser.add_argument("--checkpoint_freq", type=int, default=5, help="Checkpoint frequency")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    main(args)