from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")
results = model.track("./cleanVideo2.mp4", persist=True, save=True, device="cpu")