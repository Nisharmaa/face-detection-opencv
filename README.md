# Face Detection using OpenCV

A Python implementation of real-time face detection using OpenCV's deep learning module.

## Features
- Real-time face detection from webcam
- Deep learning-based detection using OpenCV's DNN module
- Confidence score display for each detection
- Simple and beginner-friendly code

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/face-detection-opencv.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model files:
   - [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
   - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

Place these files in the `models/` folder.

## Usage

Run the face detection script:
```bash
python face_detection.py
```

Press 'q' to quit the application.

## Technologies Used
- Python
- OpenCV
- Deep Neural Networks (DNN)

## Project Structure
```
face-detection-opencv/
├── models/           # Pre-trained model files
├── images/           # Sample images for testing
├── face_detection.py # Main implementation
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```