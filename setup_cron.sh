#!/bin/bash

# Setup script for automated training monitor cron job
# Runs every 15 minutes to check model training status

SCRIPT_DIR="/Users/owner/work"
PYTHON_PATH="/usr/bin/python3"

echo "Setting up cron job for training monitor..."

# Create the cron job entry
CRON_CMD="*/15 * * * * cd $SCRIPT_DIR && $PYTHON_PATH monitor_training.py >> cron_monitor.log 2>&1"

# Add AmazonQ coordinator job (runs 5 minutes after monitor)
AMAZONQ_CMD="5,20,35,50 * * * * cd $SCRIPT_DIR && $PYTHON_PATH amazonq_training_coordinator.py >> cron_amazonq.log 2>&1"

# Check if cron jobs already exist
(crontab -l 2>/dev/null | grep -v "monitor_training.py" | grep -v "amazonq_training_coordinator.py"; echo "$CRON_CMD"; echo "$AMAZONQ_CMD") | crontab -

echo "Cron jobs installed:"
echo "1. Training monitor - runs every 15 minutes"
echo "2. AmazonQ coordinator - runs 5 minutes after monitor"
echo ""
echo "To view current cron jobs: crontab -l"
echo "To remove cron jobs: crontab -r"
echo "To edit cron jobs: crontab -e"

# Make scripts executable
chmod +x monitor_training.py
chmod +x amazonq_training_coordinator.py

echo "Setup complete!"