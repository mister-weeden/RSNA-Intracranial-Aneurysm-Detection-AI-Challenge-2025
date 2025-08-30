#!/usr/bin/env python3
"""
AmazonQ Training Coordinator
Interfaces with AmazonQ.md files to coordinate training across all models
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('amazonq_coordinator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AmazonQCoordinator:
    def __init__(self):
        self.base_dir = Path('/Users/owner/work')
        self.models_dir = self.base_dir / 'models'
        self.models = ['deepmedic', 'dlca', 'iad', 'ihd']
        self.instruction_template = """
## AUTOMATED TRAINING INSTRUCTION
Generated: {timestamp}
Model: {model}
Current Status: {status}

### OBJECTIVE
Continue training {model} model for aneurysm detection improvement.

### CURRENT METRICS
- Best DICE Score: {best_dice}
- Training Iterations: {iterations}
- Last Checkpoint: {last_checkpoint}

### REQUIRED ACTIONS
1. Check if training/validation/testing is currently running
2. If not running, resume from last checkpoint
3. Target DICE improvement: > {target_dice}
4. Implement the following optimizations:
   - Use episodic training with memory bank (IRIS framework)
   - Apply data augmentation on query and reference images
   - Use 75%/5%/20% train/validation/test split
   - Batch size: 32, Iterations: 80K minimum
   - Optimizer: LAMB with warmup

### TRAINING PARAMETERS
```python
config = {{
    'batch_size': 32,
    'learning_rate': 1e-4,
    'optimizer': 'LAMB',
    'scheduler': 'cosine_annealing',
    'augmentation': True,
    'memory_bank': True,
    'context_ensemble': True,
    'iterations': 80000,
    'val_frequency': 1000,
    'save_frequency': 5000,
    'early_stopping_patience': 10
}}
```

### CHECKPOINT STRATEGY
- Save checkpoint when validation DICE improves
- Keep best 3 checkpoints
- Save final model after 80K iterations

### ERROR HANDLING
If training fails:
1. Check GPU memory availability
2. Reduce batch size if OOM
3. Check data loader integrity
4. Verify checkpoint compatibility

### SUCCESS CRITERIA
- DICE score > {target_dice}
- Successful validation on test set
- Model saved to checkpoints/

END OF INSTRUCTION
"""
        
    def read_model_status(self, model):
        """Read current status for a model"""
        status_file = self.base_dir / 'training_status.json'
        if status_file.exists():
            with open(status_file, 'r') as f:
                all_status = json.load(f)
                return all_status.get(model, {})
        return {
            'best_dice_score': 0.0,
            'iterations': 0,
            'last_checkpoint': None,
            'status': 'unknown'
        }
    
    def write_amazonq_instruction(self, model):
        """Write specific instruction to model's AmazonQ.md"""
        model_dir = self.models_dir / model
        amazonq_file = model_dir / 'AmazonQ.md'
        
        # Get current status
        status = self.read_model_status(model)
        
        # Calculate target DICE (1% improvement or 0.85 minimum)
        current_dice = status.get('best_dice_score', 0.0)
        target_dice = max(current_dice + 0.01, 0.85)
        
        # Generate instruction
        instruction = self.instruction_template.format(
            timestamp=datetime.now().isoformat(),
            model=model.upper(),
            status=status.get('status', 'unknown'),
            best_dice=f"{current_dice:.4f}",
            iterations=status.get('iterations', 0),
            last_checkpoint=status.get('last_checkpoint', 'None'),
            target_dice=f"{target_dice:.4f}"
        )
        
        # Write to file
        try:
            with open(amazonq_file, 'w') as f:
                f.write(instruction)
            logger.info(f"Written instruction to {amazonq_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to write instruction for {model}: {e}")
            return False
    
    def trigger_amazonq_execution(self, model):
        """Trigger AmazonQ to execute the instructions"""
        model_dir = self.models_dir / model
        
        # Create a trigger file that AmazonQ can detect
        trigger_file = model_dir / '.amazonq_trigger'
        
        trigger_content = {
            'action': 'continue_training',
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'priority': 'high',
            'auto_restart': True
        }
        
        try:
            with open(trigger_file, 'w') as f:
                json.dump(trigger_content, f, indent=2)
            logger.info(f"Created trigger for {model}")
            
            # Also try to invoke training directly
            self.start_training_process(model)
            
        except Exception as e:
            logger.error(f"Failed to create trigger for {model}: {e}")
    
    def start_training_process(self, model):
        """Start the training process for a model"""
        model_dir = self.models_dir / model
        
        # Look for training scripts
        training_scripts = [
            f'train_{model}.py',
            'train.py',
            'main.py',
            'training.py'
        ]
        
        for script in training_scripts:
            script_path = model_dir / script
            if script_path.exists():
                try:
                    cmd = f"python {script_path} --resume --auto"
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        cwd=str(model_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    logger.info(f"Started training for {model} with PID {process.pid}")
                    return process.pid
                except Exception as e:
                    logger.error(f"Failed to start {script} for {model}: {e}")
        
        logger.warning(f"No training script found for {model}")
        return None
    
    def coordinate_all_models(self):
        """Coordinate training for all models"""
        logger.info("Starting AmazonQ coordination...")
        
        results = {}
        for model in self.models:
            logger.info(f"Processing {model}...")
            
            # Write instruction
            instruction_written = self.write_amazonq_instruction(model)
            
            # Trigger execution
            if instruction_written:
                self.trigger_amazonq_execution(model)
                results[model] = 'triggered'
            else:
                results[model] = 'failed'
        
        # Write coordination summary
        summary_file = self.base_dir / 'amazonq_coordination.json'
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models': results,
            'next_check': (datetime.now().timestamp() + 900)  # 15 minutes
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Coordination complete: {results}")
        return results

def main():
    """Main execution"""
    coordinator = AmazonQCoordinator()
    
    try:
        results = coordinator.coordinate_all_models()
        
        # Print summary
        print("\nAmazonQ Coordination Summary:")
        print("-" * 40)
        for model, status in results.items():
            print(f"{model}: {status}")
        print("-" * 40)
        
    except Exception as e:
        logger.error(f"Coordination failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()