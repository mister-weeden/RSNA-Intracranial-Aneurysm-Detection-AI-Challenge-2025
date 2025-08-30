#!/bin/bash

# Installation script for the training monitor scheduler
# Works on both macOS (launchd) and Linux (cron/systemd)

echo "Installing Aneurysm Training Monitor Scheduler..."

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - using launchd"
    
    # Copy plist to LaunchAgents
    PLIST_FILE="com.aneurysm.training.monitor.plist"
    DEST_DIR="$HOME/Library/LaunchAgents"
    
    mkdir -p "$DEST_DIR"
    cp "$PLIST_FILE" "$DEST_DIR/"
    
    # Load the launch agent
    launchctl load "$DEST_DIR/$PLIST_FILE"
    
    echo "LaunchAgent installed and started"
    echo "To check status: launchctl list | grep aneurysm"
    echo "To stop: launchctl unload ~/Library/LaunchAgents/$PLIST_FILE"
    echo "To remove: launchctl unload ~/Library/LaunchAgents/$PLIST_FILE && rm ~/Library/LaunchAgents/$PLIST_FILE"
    
else
    echo "Detected Linux/Unix - using cron"
    
    # Install cron job
    SCRIPT_DIR="$(pwd)"
    PYTHON_PATH="$(which python3)"
    
    # Create cron entries
    CRON_MONITOR="*/15 * * * * cd $SCRIPT_DIR && $PYTHON_PATH monitor_training.py >> cron_monitor.log 2>&1"
    CRON_AMAZONQ="5,20,35,50 * * * * cd $SCRIPT_DIR && $PYTHON_PATH amazonq_training_coordinator.py >> cron_amazonq.log 2>&1"
    
    # Add to crontab
    (crontab -l 2>/dev/null | grep -v "monitor_training.py" | grep -v "amazonq_training_coordinator.py"; 
     echo "$CRON_MONITOR"; 
     echo "$CRON_AMAZONQ") | crontab -
    
    echo "Cron jobs installed"
    echo "To check: crontab -l"
    echo "To remove: crontab -e (then delete the lines)"
fi

# Make scripts executable
chmod +x monitor_training.py
chmod +x amazonq_training_coordinator.py

echo ""
echo "Installation complete!"
echo "The monitor will check all 4 models every 15 minutes and:"
echo "1. Check if training/validation/testing is running"
echo "2. Monitor checkpoint improvements"
echo "3. Restart training if needed via AmazonQ instructions"
echo "4. Target DICE score improvement for aneurysm detection"