# Sus-Detector: Intelligent Surveillance System

## Overview

Sus-Detector is an intelligent surveillance system that uses computer vision and machine learning to detect suspicious activities and recognize faces. The system can automatically control access (door opening/closing) based on face recognition and alert users to potential security threats.

## Features

- Real-time video surveillance with YOLO object detection
- Face recognition to identify known individuals
- Automated door access control system
- Integration with Firebase for remote monitoring and control
- Suspicious activity detection and alerting

## Requirements

- Python 3.x
- pip (Python package manager)
- Webcam or video source
- Internet connection for Firebase integration

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sus-detector.git
cd sus-detector
```

### 2. Set Up a Virtual Environment

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS and Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Known Faces

Place images of known faces in the `known_faces` directory. Name each image file with the person's name (e.g., `john_doe.jpg`).

## Usage

### Running the Application

```bash
python program.py
```

### Configuration

You can modify various settings in the `program.py` file:

#### Change YOLO Model

To use a different YOLO model, modify the model initialization:

```python
self.model = YOLO('model_1.pt')
```

#### Firebase Configuration

Update Firebase credentials in the `program.py` file to connect to your own Firebase project.

## Directory Structure

```
sus-detector/
├── program.py            # Main application code
├── requirements.txt      # Python dependencies
├── known_faces/          # Directory for known face images
│   ├── person1.jpg
│   └── person2.jpg
├── models/               # YOLO models directory
└── README.md             # This documentation
```

## How It Works

The system operates using multiple processing threads:
1. Video capture from camera input
2. Object detection using YOLO model
3. Face recognition against known faces database
4. Door control system based on recognition results
5. Firebase integration for remote monitoring and control

## Troubleshooting

- **Face Recognition Issues**: Ensure good lighting in images and during detection
- **Model Performance**: Try different YOLO models if detection accuracy is insufficient
- **Camera Access**: Check camera permissions if the video feed doesn't initialize

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- YOLOv8 for object detection
- face_recognition library for facial recognition
- Firebase for remote database functionality

## Contact
For any questions or issues, please contact
- [Duy Nguyen](mailto:nguyenducduypc160903@gmail.com)