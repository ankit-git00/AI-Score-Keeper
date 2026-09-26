import cv2, json
import numpy as np

data = json.load(open("court_polygon.json"))
poly = np.array(data["polygon"], dtype=np.int32)

cap = cv2.VideoCapture("cleanVideo2.mp4")
for t in [2, 30, 80, 150]:  # check several timestamps
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        continue
    cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(f"check_{t}.jpg", frame)
cap.release()