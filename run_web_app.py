#!/usr/bin/env python3
"""
Web Application Runner for Live Parking Space Detection System

This script runs the Flask web application that provides:
- Real-time parking space statistics
- Live camera feed display
- Detection results visualization
- Region breakdown
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("Starting Live Parking Space Detection Web Application")
    print("=" * 60)
    print("Features:")
    print("- Real-time parking statistics")
    print("- Live camera feed display")
    print("- Detection results visualization")
    print("- Region breakdown")
    print("- Modern responsive web interface")
    print("=" * 60)
    print("Access the web interface at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Import and run the Flask app
        from app import app, init_detection_system
        # Ensure detection system starts immediately when launching the web app
        try:
            init_detection_system()
        except Exception as _e:
            print(f"Warning: detection init failed on startup: {_e}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
