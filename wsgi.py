#!/usr/bin/env python3
"""
WSGI entry point for Azure App Service deployment
This file is required for GitHub deployment to Azure App Service
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask application
from app import app

# This is the WSGI application object that Azure will use
application = app

if __name__ == "__main__":
    # This will only run if the script is executed directly
    # In production, Azure will use the 'application' object above
    app.run(host='0.0.0.0', port=8000, debug=False)
