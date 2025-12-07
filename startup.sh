#!/bin/bash

# Azure App Service startup script for GitHub deployment
# This script is used when deploying from GitHub to Azure App Service

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${HOME}/site/wwwroot"

# Navigate to the application directory
cd ${HOME}/site/wwwroot

# Install dependencies if requirements.txt exists
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt --user
fi

# Start the application with gunicorn
echo "Starting application with gunicorn..."
exec gunicorn --bind=0.0.0.0 --timeout 600 --workers 2 --threads 4 wsgi:application
