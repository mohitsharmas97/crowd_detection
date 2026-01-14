import cv2
from ultralytics import YOLO
import sys
import os

# Path to the video file
video_path = "c:/Users/Mohit Sharma/Desktop/crowd_detection/static/uploads/video_20260114_145822_VID-20250923-WA0008_clip_001.mp4"

if not os.path.exists(video_path):
    # Try finding any mp4 in uploads if specific one not found (for robustness)
    print(f"File not found: {video_path}")
    uploads = "c:/Users/Mohit Sharma/Desktop/crowd_detection/static/uploads"
    files = [f for f in os.listdir(uploads) if f.endswith('.mp4')]
    if files:
        video_path = os.path.join(uploads, files[0])
        print(f"Using alternative video: {video_path}")
    else:
        print("No video files found.")
        sys.exit(1)

print(f"Analyzing video: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    sys.exit(1)

# Read a few frames to get past any black start frames
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break

if not ret:
    print("Could not read frame")
    sys.exit(1)

print(f"Frame Shape: {frame.shape}")
height, width = frame.shape[:2]

# Load model
try:
    model = YOLO('yolov8m.pt')
    print("Model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Run detection
print("Running detection with imgsz=1280 and conf=0.15...")
results = model(frame, verbose=True, imgsz=1280, conf=0.15)[0]

print("-" * 30)
print(f"Total Detections: {len(results.boxes)}")
if len(results.boxes) > 0:
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {cls}, Conf: {conf:.4f}")

# Check with lower threshold
print("-" * 30)
print("Checking with lower threshold (0.1)...")
results_low = model(frame, conf=0.1, verbose=False)[0]
person_count = 0
for box in results_low.boxes:
    if int(box.cls[0]) == 0:
        person_count += 1
print(f"Person Count at 0.1 conf: {person_count}")

cap.release()
