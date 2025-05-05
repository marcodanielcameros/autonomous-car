This project simulates an autonomous car in the Webots robot simulator, using a camera for real-time lane detection and control. The system processes camera input through OpenCV to detect road lanes using image processing techniques such as grayscale conversion, blurring, Canny edge detection, region of interest masking, and Hough line transformation.

🔍 Features
Real-time lane detection using OpenCV

Steering control based on lane center offset (proportional control logic)

Visualization of processed images within Webots displays

Manual override via keyboard controls

Option to capture camera images with timestamp

📦 Key Technologies
Webots Simulation Environment

Python

OpenCV (cv2)

NumPy

🎮 Controls
Arrow keys to control steering and speed

A key to save a camera snapshot
