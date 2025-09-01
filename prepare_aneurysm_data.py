#!/usr/bin/env python3
"""
Prepare aneurysm data for MedicalNet training
Converts DICOM to NIfTI and organizes data for training
"""

import os
import sys
import subprocess
import shutil
import json
import numpy as np
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AneurysmDataPreparer:
    def __init__(self):
        self.base_dir = Path('/Users/owner/work')
        self.data_dir = self.base_dir / 'data'
        self.dcm2niix = self.data_dir / 'dcm2niix'
        self.output_dir = self.base_dir / 'aneurysm_dataset'
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        # Create output directories
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def find_dicom_directories(self):
        """Find all DICOM directories in data folder"""
        dicom_dirs = []
        
        # Look for directories with numeric patterns (typical DICOM structure)
        for item in self.data_dir.iterdir():
            if item.is_dir() and '1.2.826' in str(item):
                dicom_dirs.append(item)
        
        logger.info(f"Found {len(dicom_dirs)} DICOM directories")
        return dicom_dirs
    
    def convert_dicom_to_nifti(self, dicom_dir, output_name):
        """Convert DICOM directory to NIfTI using dcm2niix"""
        try:
            cmd = [
                str(self.dcm2niix),
                '-z', 'y',  # Compress
                '-f', output_name,  # Output filename
                '-o', str(self.output_dir / 'temp'),  # Output directory
                str(dicom_dir)
            ]
            
            # Create temp directory
            (self.output_dir / 'temp').mkdir(exist_ok=True)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully converted {dicom_dir.name}")
                return True
            else:
                logger.error(f"Failed to convert {dicom_dir.name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error converting {dicom_dir}: {e}")
            return False
    
    def organize_nifti_files(self):
        """Organize existing NIfTI files into train/val/test splits"""
        # Find all NIfTI files
        nifti_files = list(self.data_dir.glob('*.nii')) + list(self.data_dir.glob('*.nii.gz'))
        
        # Filter for aneurysm-related files (excluding segmentation masks for now)
        data_files = [f for f in nifti_files if '_cowseg' not in f.name]
        seg_files = [f for f in nifti_files if '_cowseg' in f.name]
        
        logger.info(f"Found {len(data_files)} data files and {len(seg_files)} segmentation files")
        
        # Create 75/5/20 split
        random.shuffle(data_files)
        n_total = len(data_files)
        n_train = int(n_total * 0.75)
        n_val = int(n_total * 0.05)
        
        train_files = data_files[:n_train]
        val_files = data_files[n_train:n_train + n_val]
        test_files = data_files[n_train + n_val:]
        
        # Copy files to respective directories
        for files, target_dir, split_name in [
            (train_files, self.train_dir, 'train'),
            (val_files, self.val_dir, 'val'),
            (test_files, self.test_dir, 'test')
        ]:
            for i, file in enumerate(files):
                # Copy data file
                new_name = f"aneurysm_{split_name}_{i:04d}.nii"
                if file.suffix == '.gz':
                    new_name += '.gz'
                
                target_path = target_dir / new_name
                shutil.copy2(file, target_path)
                
                # Look for corresponding segmentation
                seg_name = file.stem.replace('.nii', '') + '_cowseg.nii'
                seg_path = self.data_dir / seg_name
                if seg_path.exists():
                    seg_target = target_dir / f"aneurysm_{split_name}_{i:04d}_seg.nii"
                    shutil.copy2(seg_path, seg_target)
                    logger.info(f"Copied {file.name} with segmentation to {split_name}")
                else:
                    logger.info(f"Copied {file.name} to {split_name} (no segmentation)")
        
        return len(train_files), len(val_files), len(test_files)
    
    def create_dataset_json(self):
        """Create dataset configuration JSON for training"""
        dataset_config = {
            "name": "RSNA_Aneurysm_Detection",
            "description": "RSNA Intracranial Aneurysm Detection Dataset",
            "modality": "MRI",
            "labels": {
                "0": "background",
                "1": "aneurysm"
            },
            "numTraining": len(list(self.train_dir.glob('*.nii*'))),
            "numValidation": len(list(self.val_dir.glob('*.nii*'))),
            "numTest": len(list(self.test_dir.glob('*.nii*'))),
            "training": [str(f) for f in self.train_dir.glob('*.nii*') if '_seg' not in f.name],
            "validation": [str(f) for f in self.val_dir.glob('*.nii*') if '_seg' not in f.name],
            "test": [str(f) for f in self.test_dir.glob('*.nii*') if '_seg' not in f.name]
        }
        
        config_path = self.output_dir / 'dataset.json'
        with open(config_path, 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        logger.info(f"Created dataset configuration at {config_path}")
        return config_path
    
    def prepare_data(self):
        """Main data preparation pipeline"""
        logger.info("Starting aneurysm data preparation...")
        
        # Step 1: Organize existing NIfTI files
        n_train, n_val, n_test = self.organize_nifti_files()
        logger.info(f"Data split: Train={n_train}, Val={n_val}, Test={n_test}")
        
        # Step 2: Convert any remaining DICOM directories
        dicom_dirs = self.find_dicom_directories()
        if dicom_dirs:
            logger.info(f"Converting {len(dicom_dirs)} DICOM directories...")
            for i, dicom_dir in enumerate(dicom_dirs[:10]):  # Convert first 10 for now
                self.convert_dicom_to_nifti(dicom_dir, f"aneurysm_extra_{i:04d}")
        
        # Step 3: Create dataset configuration
        config_path = self.create_dataset_json()
        
        # Step 4: Generate summary
        self.generate_summary()
        
        return config_path
    
    def generate_summary(self):
        """Generate data preparation summary"""
        summary = f"""
Aneurysm Data Preparation Summary
=====================================
Output Directory: {self.output_dir}

Data Split (75/5/20):
- Training: {len(list(self.train_dir.glob('*.nii*')))} files
- Validation: {len(list(self.val_dir.glob('*.nii*')))} files  
- Testing: {len(list(self.test_dir.glob('*.nii*')))} files

Files organized in:
- {self.train_dir}
- {self.val_dir}
- {self.test_dir}

Dataset configuration: {self.output_dir / 'dataset.json'}
"""
        
        logger.info(summary)
        
        # Save summary
        with open(self.output_dir / 'preparation_summary.txt', 'w') as f:
            f.write(summary)

def main():
    preparer = AneurysmDataPreparer()
    config_path = preparer.prepare_data()
    print(f"Data preparation complete! Configuration saved to: {config_path}")

if __name__ == "__main__":
    main()