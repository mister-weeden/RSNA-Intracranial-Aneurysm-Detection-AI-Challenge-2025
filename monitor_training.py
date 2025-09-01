#!/usr/bin/env python3
"""
Automated Training Monitor for Aneurysm Detection Models
Monitors 4 models: deepmedic, dlca, iad, ihd
Checks training status every 15 minutes and restarts if needed
"""

import os
import sys
import subprocess
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ModelTrainingMonitor:
    def __init__(self):
        self.base_dir = Path('/Users/owner/work')
        self.models = ['deepmedic', 'dlca', 'iad', 'ihd']
        self.model_dirs = {
            'deepmedic': self.base_dir / 'models' / 'deepmedic',
            'dlca': self.base_dir / 'models' / 'dlca',
            'iad': self.base_dir / 'models' / 'iad',
            'ihd': self.base_dir / 'models' / 'ihd'
        }
        self.checkpoint_dirs = {
            model: self.model_dirs[model] / 'checkpoints' 
            for model in self.models
        }
        self.status_file = self.base_dir / 'training_status.json'
        self.load_status()
        
    def load_status(self):
        """Load previous training status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {
                model: {
                    'last_checkpoint': None,
                    'best_auc_score': 0.0,
                    'sensitivity': 0.0,
                    'specificity': 0.0,
                    'false_positive_rate': 1.0,
                    'true_negative_rate': 0.0,
                    'training_pid': None,
                    'last_check': None,
                    'iterations': 0,
                    'status': 'idle'
                } for model in self.models
            }
    
    def save_status(self):
        """Save current training status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)
    
    def check_process_running(self, model):
        """Check if training process is running for a model"""
        # Use ps command to check for running processes
        try:
            cmd = f"ps aux | grep -i {model} | grep -E 'train|validation|test' | grep -v grep"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                # Extract PID from first matching process
                lines = result.stdout.strip().split('\n')
                if lines:
                    # PID is typically the second field in ps aux output
                    parts = lines[0].split()
                    if len(parts) > 1:
                        return int(parts[1])
        except Exception as e:
            logger.warning(f"Error checking process for {model}: {e}")
        
        return None
    
    def get_latest_checkpoint(self, model):
        """Get the latest checkpoint for a model"""
        checkpoint_dir = self.checkpoint_dirs[model]
        if not checkpoint_dir.exists():
            return None, 0.0
        
        checkpoints = list(checkpoint_dir.glob('*.pth')) + \
                     list(checkpoint_dir.glob('*.pt')) + \
                     list(checkpoint_dir.glob('*.ckpt'))
        
        if not checkpoints:
            return None, 0.0
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        # Try to extract AUC score from filename or checkpoint
        auc_score = self.extract_auc_score(latest)
        
        return str(latest), auc_score
    
    def extract_auc_score(self, checkpoint_path):
        """Extract AUC score from checkpoint filename or content"""
        # Try to extract from filename first
        filename = checkpoint_path.name
        match = re.search(r'auc[_-]?(\d+\.?\d*)', filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Try to load checkpoint and get score
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                for key in ['auc_score', 'best_auc', 'val_auc', 'test_auc', 'roc_auc']:
                    if key in checkpoint:
                        return float(checkpoint[key])
        except:
            pass
        
        return 0.0
    
    def check_improvement(self, model):
        """Check if model has improved"""
        latest_checkpoint, auc_score = self.get_latest_checkpoint(model)
        
        if latest_checkpoint and auc_score > self.status[model]['best_auc_score']:
            logger.info(f"{model}: New best AUC score: {auc_score:.4f}")
            self.status[model]['best_auc_score'] = auc_score
            self.status[model]['last_checkpoint'] = latest_checkpoint
            return True
        return False
    
    def restart_training(self, model):
        """Restart training for a model using AmazonQ instructions"""
        logger.info(f"Restarting training for {model}")
        
        amazonq_file = self.model_dirs[model] / 'AmazonQ.md'
        
        # Read AmazonQ instructions
        if amazonq_file.exists():
            with open(amazonq_file, 'r') as f:
                instructions = f.read()
            
            # Extract training command from AmazonQ.md
            train_command = self.extract_training_command(instructions, model)
            
            if train_command:
                # Execute training command
                try:
                    process = subprocess.Popen(
                        train_command,
                        shell=True,
                        cwd=str(self.model_dirs[model]),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    self.status[model]['training_pid'] = process.pid
                    self.status[model]['status'] = 'training'
                    logger.info(f"{model}: Started training with PID {process.pid}")
                    
                    # Update AmazonQ.md with continuation instructions
                    self.update_amazonq_instructions(model)
                    
                except Exception as e:
                    logger.error(f"{model}: Failed to start training - {e}")
        else:
            logger.warning(f"{model}: No AmazonQ.md file found")
    
    def extract_training_command(self, instructions, model):
        """Extract training command from AmazonQ instructions"""
        # Look for python training commands in the instructions
        patterns = [
            r'python\s+train.*\.py',
            r'python3\s+train.*\.py',
            r'python.*training.*\.py',
            f'python.*{model}.*train'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, instructions, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(0)
        
        # Default training commands for each model
        default_commands = {
            'deepmedic': 'python train_deepmedic.py --resume --epochs 100',
            'dlca': 'python train_dlca.py --resume --batch_size 32',
            'iad': 'python train_iad.py --resume --iterations 80000',
            'ihd': 'python train_ihd.py --resume --checkpoint latest'
        }
        
        return default_commands.get(model)
    
    def update_amazonq_instructions(self, model):
        """Update AmazonQ.md with continuation instructions"""
        amazonq_file = self.model_dirs[model] / 'AmazonQ.md'
        
        continuation_note = f"""

## Automated Training Continuation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The training monitor has detected that {model} training has stopped.
Current best AUC score: {self.status[model]['best_auc_score']:.4f}
Last checkpoint: {self.status[model]['last_checkpoint']}

### Instructions for continuation:
1. Resume from the latest checkpoint
2. Continue training until improvement in aneurysm detection AUC score
3. Target: Achieve AUC score > {self.status[model]['best_auc_score'] + 0.01:.4f}
4. Monitor: Sensitivity, Specificity, False Positive Rate
4. Save checkpoint when improvement is detected
5. Implement early stopping if no improvement after 10 epochs

### Training parameters to adjust if needed:
- Learning rate decay if plateauing
- Increase augmentation if overfitting
- Adjust batch size for memory optimization
"""
        
        try:
            with open(amazonq_file, 'a') as f:
                f.write(continuation_note)
            logger.info(f"{model}: Updated AmazonQ.md with continuation instructions")
        except Exception as e:
            logger.error(f"{model}: Failed to update AmazonQ.md - {e}")
    
    def monitor_all_models(self):
        """Main monitoring function for all models"""
        logger.info("Starting training monitor check...")
        
        for model in self.models:
            logger.info(f"Checking {model}...")
            
            # Check if process is running
            pid = self.check_process_running(model)
            
            if pid:
                logger.info(f"{model}: Training process running (PID: {pid})")
                self.status[model]['training_pid'] = pid
                self.status[model]['status'] = 'running'
                
                # Check for improvements
                if self.check_improvement(model):
                    self.status[model]['iterations'] += 1
            else:
                logger.warning(f"{model}: No training process detected")
                self.status[model]['status'] = 'stopped'
                
                # Check if we should restart
                if self.should_restart(model):
                    self.restart_training(model)
            
            self.status[model]['last_check'] = datetime.now().isoformat()
        
        self.save_status()
        logger.info("Monitor check completed")
    
    def should_restart(self, model):
        """Determine if training should be restarted"""
        # Don't restart if recently checked
        if self.status[model]['last_check']:
            last_check = datetime.fromisoformat(self.status[model]['last_check'])
            if (datetime.now() - last_check).seconds < 300:  # 5 minutes cooldown
                return False
        
        # Restart if no checkpoint exists or AUC score is below threshold
        if self.status[model]['best_auc_score'] < 0.90:  # Target 90% AUC
            return True
        
        # Restart if iterations are below target
        if self.status[model]['iterations'] < 80000:
            return True
        
        return False
    
    def generate_report(self):
        """Generate training status report"""
        report = [
            "\n" + "="*60,
            f"Training Status Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*60
        ]
        
        for model in self.models:
            status = self.status[model]
            report.extend([
                f"\n{model.upper()}:",
                f"  Status: {status['status']}",
                f"  Best AUC: {status.get('best_auc_score', 0.0):.4f}",
                f"  Sensitivity: {status.get('sensitivity', 0.0):.4f}",
                f"  Specificity: {status.get('specificity', 0.0):.4f}",
                f"  FPR: {status.get('false_positive_rate', 1.0):.4f}",
                f"  Iterations: {status['iterations']}",
                f"  Last Check: {status['last_check']}",
                f"  PID: {status['training_pid']}"
            ])
        
        report_text = "\n".join(report)
        logger.info(report_text)
        
        # Save report to file
        with open(self.base_dir / 'training_report.txt', 'w') as f:
            f.write(report_text)
        
        return report_text

def main():
    """Main execution function"""
    monitor = ModelTrainingMonitor()
    
    try:
        # Run monitoring
        monitor.monitor_all_models()
        
        # Generate report
        monitor.generate_report()
        
    except Exception as e:
        logger.error(f"Monitor failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()