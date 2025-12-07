# Parking Space Detection System


A real-time parking space detection system using computer vision and deep learning. This system uses YOLO (You Only Look Once) segmentation model to detect vehicles in parking lots and provides a web-based interface for monitoring parking availability.

image.png

## Features

- Real-time Vehicle Detection: Uses YOLOv8 segmentation model to detect vehicles (cars, motorcycles, trucks) in parking spaces
- Multi-Parking Lot Support: Monitor multiple parking lots simultaneously with separate camera feeds
- Live Camera Feed: Real-time video feed from IP cameras (MJPEG, RTSP, HTTP)
- Interactive Calibration Tool: Web-based and desktop tools for defining parking regions
- Statistics Dashboard: Real-time statistics showing total, empty, and filled parking spaces
- Historical Data: Database storage for parking statistics and session history
- Admin Panel: Secure admin interface for managing parking lots and calibration
- RESTful API: JSON API endpoints for integration with other systems
- Azure Deployment Ready: Optimized for deployment on Azure App Service

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Access to IP camera feed (MJPEG, RTSP, or HTTP snapshot)
- (Optional) CUDA-enabled GPU for faster inference

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rameez-anwar/Parking-Space-Detection-CV-Project.git
cd Parking-Space-Detection-CV-Project
```

### 2. Install Dependencies


```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

```bash
python run_web_app.py
```

Or directly:

```bash
python app.py
```

The web interface will be available at: `http://localhost:5000`

### Running the Calibration Tool

For desktop calibration tool (Tkinter-based):

```bash
python calibration_tool.py
```

### Default Credentials

- Username: `admin`
- Password: `pass`

IMPORTANT: Change these credentials in production! Edit `app.py` and update `ADMIN_USERNAME` and `ADMIN_PASSWORD`.

## Project Structure

```
ParkingSpace/
├── app.py                      # Main Flask application
├── database.py                 # SQLite database operations
├── calibration_tool.py         # Desktop calibration tool (Tkinter)
├── run_web_app.py             # Web application runner
├── wsgi.py                    # WSGI entry point for Azure
├── requirements.txt           # Python dependencies
├── model.pt                   # YOLO model weights
├── parking_data.db            # SQLite database (auto-created)
├── regions.json               # Parking region definitions
├── templates/                 # HTML templates
│   ├── dashboard.html
│   ├── admin_panel.html
│   ├── admin_calibration.html
│   ├── admin_login.html
│   ├── main_login.html
│   └── index.html
└── README.md            
```

## Configuration

### Setting Up Parking Lots

1. Log in to the admin panel at `/admin`
2. Add a new parking lot with:
   - Name: Display name for the parking lot
   - Camera URL: IP camera feed URL (e.g., `http://170.249.152.2:8080/video.mjpg`)
   - Regions File: JSON file containing parking region definitions

### Calibrating Parking Regions

1. Navigate to `/admin/calibration/<lot_id>` for a specific parking lot
2. Capture a frame from the camera
3. Draw polygonal regions around parking spaces
4. Save regions to the JSON file
5. The system will automatically restart detection with new regions

### Supported Camera Formats

- MJPEG Stream: `http://ip:port/video.mjpg`
- RTSP Stream: `rtsp://ip:port/stream`
- HTTP Snapshot: `http://ip:port/snapshot.jpg`

## API Endpoints

### Public Endpoints

- `GET /api/parking-lots` - Get list of all parking lots
- `GET /api/statistics?lot_id=<id>` - Get parking statistics (optional lot_id parameter)
- `GET /api/live-image/<lot_id>` - Get current live camera image
- `GET /api/processed-image/<lot_id>` - Get processed image with detections
- `GET /api/history?limit=50` - Get historical statistics
- `GET /healthz` - Health check endpoint

### Admin Endpoints (Requires Authentication)

- `GET /admin` - Admin panel
- `GET /admin/calibration/<lot_id>` - Calibration tool for parking lot
- `POST /admin/api/parking-lots` - Add new parking lot
- `PUT /admin/api/parking-lots/<lot_id>` - Update parking lot
- `DELETE /admin/api/parking-lots/<lot_id>` - Delete parking lot
- `POST /admin/api/save-regions/<lot_id>` - Save parking regions
- `GET /admin/api/capture-frame/<lot_id>` - Capture frame for calibration

## Database Schema

The system uses SQLite database (`parking_data.db`) with the following tables:

- `parking_lots`: Stores parking lot configurations
- `detection_sessions`: Tracks detection sessions
- `parking_statistics`: Stores detection results and statistics
- `system_config`: System configuration settings


## How It Works

1. Camera Feed: System connects to IP camera feed (MJPEG/RTSP/HTTP)
2. Frame Capture: Captures frames every 15 seconds
3. Vehicle Detection: YOLO model detects vehicles in the frame
4. Region Analysis: Checks each defined parking region for vehicle coverage
5. Occupancy Calculation: Determines if a space is occupied (>=30% vehicle coverage)
6. Statistics Update: Updates database with current parking statistics
7. Web Display: Real-time dashboard shows current availability

## Troubleshooting

### Camera Connection Issues

- Verify camera URL is accessible
- Check network connectivity
- Ensure camera supports MJPEG/RTSP/HTTP snapshot

### Model Loading Errors

- Ensure PyTorch is installed correctly
- Check model file exists or can be downloaded
- Verify sufficient disk space for model weights

### Detection Not Working

- Verify regions are properly defined in JSON file
- Check camera feed is active and accessible
- Review console logs for error messages

## License

This project is part of a Computer Vision Lab course project.

## Authors

- Rameez Anwar - GitHub: https://github.com/rameez-anwar

## Acknowledgments

- Ultralytics (https://github.com/ultralytics/ultralytics) for YOLOv8
- OpenCV (https://opencv.org/) for computer vision operations
- Flask (https://flask.palletsprojects.com/) for web framework

## Support

For issues and questions, please open an issue on the GitHub repository.

---

Note: This is an academic project for Computer Vision Lab (Semester 7). For production use, ensure proper security measures, change default credentials, and implement additional error handling.

