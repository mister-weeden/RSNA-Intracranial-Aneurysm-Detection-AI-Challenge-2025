# Aneurysm Detection Training Monitor System

## Overview
Automated monitoring system that checks all 4 models (deepmedic, dlca, iad, ihd) every 15 minutes to ensure continuous training progress for aneurysm detection improvement.

## Components

### 1. monitor_training.py
Main monitoring script that:
- Checks if training/validation/testing processes are running for each model
- Monitors checkpoint improvements and DICE scores
- Automatically restarts training if processes stop
- Updates AmazonQ.md files with continuation instructions
- Tracks best DICE scores and training iterations

### 2. amazonq_training_coordinator.py
Coordination script that:
- Generates detailed training instructions for AmazonQ
- Creates trigger files for automated execution
- Manages training parameters and optimization strategies
- Implements IRIS framework recommendations

### 3. Scheduler (LaunchAgent/Cron)
- Runs monitor_training.py every 15 minutes
- Runs amazonq_training_coordinator.py 5 minutes after monitor
- Automatically starts on system boot (macOS LaunchAgent)

## Installation

```bash
# Make scripts executable
chmod +x install_scheduler.sh

# Install the scheduler
./install_scheduler.sh
```

## Usage

### Manual Testing
```bash
# Test monitoring script
python3 monitor_training.py

# Test AmazonQ coordinator
python3 amazonq_training_coordinator.py
```

### Check Status
```bash
# View current training status
cat training_status.json

# View training report
cat training_report.txt

# Check scheduler status (macOS)
launchctl list | grep aneurysm

# View logs
tail -f training_monitor.log
tail -f amazonq_coordinator.log
```

### Managing the Scheduler

#### macOS (LaunchAgent)
```bash
# Stop scheduler
launchctl unload ~/Library/LaunchAgents/com.aneurysm.training.monitor.plist

# Start scheduler
launchctl load ~/Library/LaunchAgents/com.aneurysm.training.monitor.plist

# Remove scheduler
launchctl unload ~/Library/LaunchAgents/com.aneurysm.training.monitor.plist
rm ~/Library/LaunchAgents/com.aneurysm.training.monitor.plist
```

#### Linux (Cron)
```bash
# View cron jobs
crontab -l

# Edit cron jobs
crontab -e

# Remove all cron jobs
crontab -r
```

## Training Goals

Each model targets:
- Minimum DICE score: 0.85 (85%)
- Training iterations: 80,000
- Batch size: 32
- Optimizer: LAMB
- Data split: 75%/5%/20% (train/validation/test)

## AmazonQ Integration

The system automatically updates AmazonQ.md files in each model directory with:
1. Current training status
2. Best DICE score achieved
3. Instructions for continuation
4. Target metrics for improvement
5. Training parameter adjustments

## Files Generated

- `training_status.json` - Current status of all models
- `training_report.txt` - Human-readable status report
- `training_monitor.log` - Monitor script logs
- `amazonq_coordinator.log` - Coordinator script logs
- `launchd_monitor.log` - Scheduler output (macOS)
- `models/*/AmazonQ.md` - Updated training instructions
- `models/*/.amazonq_trigger` - Trigger files for automation

## Monitoring Logic

Every 15 minutes:
1. Check if each model has a running training process
2. If not running:
   - Check if DICE score < 0.85 or iterations < 80K
   - If yes, restart training with updated parameters
   - Update AmazonQ.md with continuation instructions
3. If running:
   - Check for new checkpoints
   - Compare DICE scores for improvement
   - Update status tracking

## Success Criteria

Training continues until:
- DICE score > 0.85 for aneurysm detection
- 80,000 iterations completed
- Validation on test set successful
- Checkpoint saved with improvement

## Troubleshooting

### Process not starting
- Check Python path in scripts
- Verify model directories exist
- Check for training scripts in model folders

### Scheduler not running
- Verify LaunchAgent is loaded (macOS)
- Check crontab entries (Linux)
- Review log files for errors

### No improvement in DICE scores
- System will automatically adjust:
  - Learning rate decay
  - Data augmentation
  - Batch size optimization
  - Early stopping after 10 epochs without improvement