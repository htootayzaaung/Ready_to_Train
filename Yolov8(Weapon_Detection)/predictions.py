from ultralytics import YOLO

import cv2
model = YOLO("runs/detect/train6/weights/best.pt")
model.predict(source="0", show=True, conf=0.40)
