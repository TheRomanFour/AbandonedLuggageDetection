# AbandonedLuggageDetection
## YOLOv8 Object Tracking and Abandonment Detection
This project implements a real-time object tracking and abandonment detection system using the YOLOv8 object detection model. The primary focus is on detecting and tracking persons and suitcases in a video feed and identifying potentially abandoned suitcases based on their movement and proximity to people. This project is part of my master's thesis and features custom-trained YOLOv8 models that have been specifically trained on custom datasets to enhance detection accuracy for the target objects.

## Features
Custom Trained Models: Includes YOLOv8 models trained on specialized datasets as part of my master's thesis, improving detection accuracy for persons and suitcases.
Real-Time Object Tracking: Detects and tracks persons and suitcases in each video frame using the YOLOv8 model.
Proximity-Based Connections: Automatically links suitcases with the nearest person based on proximity within the video frame.
## Abandonment Detection: Flags a suitcase as "Potentially Abandoned" if:
It has been stationary for more than 5 seconds.
No person is detected within a certain radius (300 units by default).
Dynamic Annotations: Displays real-time annotations, including bounding boxes, object IDs, and distances between associated objects.
## Requirements
* Python 3.7+
* OpenCV
* NumPy
* Ultralytics YOLO

## Dataset links:
* https://app.roboflow.com/cars-0jbgu/luggage-person-detection-airport (Airport)
* https://app.roboflow.com/cars-0jbgu/luggage-detection-axdmv/1 (City of Rijeka, Korzo)
