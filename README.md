# Basic ASL Gesture Translator (OpenCV)

This project demonstrates a beginner-level Computer Vision system that recognizes basic American Sign Language (ASL) gestures using a webcam.

## How it Works
- Captures live video using OpenCV
- Segments the hand using HSV skin color detection
- Finds the largest contour as the hand
- Computes convex hull and convexity defects
- Counts raised fingers using rule-based logic
- Maps finger count to ASL letters (A–E)

## Gesture Mapping
1 Finger → A  
2 Fingers → B  
3 Fingers → C  
4 Fingers → D  
5 Fingers → E  

## Technologies Used
- Python
- OpenCV
- NumPy

## How to Run
```bash
pip install opencv-python numpy
python hand_detection.py
