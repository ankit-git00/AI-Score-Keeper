"""Run YOLO-pose tracking on a video, keep only people inside the marked court
polygon, and write a new video with just those detections drawn on.

Usage:
    python annotate_court_players.py cleanVideo1.mp4 court_polygon.json --out annotated.avi

Requires: pip install ultralytics opencv-python
"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

from court_filter import CourtFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("polygon", help="court_polygon.json from mark_court.py")
    ap.add_argument("--out", default="annotated.avi")
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--draw-polygon", action="store_true", default=True,
                     help="also draw the court boundary on every frame")
    a = ap.parse_args()

    cf = CourtFilter(a.polygon)
    model = YOLO(a.model)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(a.out, fourcc, fps, (w, h))

    # Pose skeleton connections (COCO 17-keypoint order used by YOLOv8-pose)
    SKELETON = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
    ]

    n_frames = 0
    n_kept_boxes = 0

    results = model.track(a.video, persist=True, device=a.device, stream=True, verbose=False)
    for r in results:
        frame = r.orig_img.copy()
        frame_size = (r.orig_shape[1], r.orig_shape[0])  # (w, h)

        if a.draw_polygon:
            poly = cf.polygon.copy().astype(np.float64)
            if frame_size != cf.frame_size:
                sx = frame_size[0] / cf.frame_size[0]
                sy = frame_size[1] / cf.frame_size[1]
                poly[:, 0] *= sx
                poly[:, 1] *= sy
            cv2.polylines(frame, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.tolist()
            xyxy = r.boxes.xyxy.tolist()
            kpts = r.keypoints.xy.tolist() if r.keypoints is not None else [None] * len(ids)

            for tid, box, kp in zip(ids, xyxy, kpts):
                x1, y1, x2, y2 = box
                if not cf.is_on_court((x1, y1, x2, y2), frame_size=frame_size):
                    continue  # drop: outside the marked court region
                n_kept_boxes += 1

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {int(tid)}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if kp:
                    pts = [(int(x), int(y)) for x, y in kp]
                    for x, y in pts:
                        if x > 0 and y > 0:
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                    for i, j in SKELETON:
                        if i < len(pts) and j < len(pts):
                            xi, yi = pts[i]
                            xj, yj = pts[j]
                            if xi > 0 and yi > 0 and xj > 0 and yj > 0:
                                cv2.line(frame, (xi, yi), (xj, yj), (255, 128, 0), 2)

        writer.write(frame)
        n_frames += 1
        if n_frames % 100 == 0:
            print(f"  processed {n_frames} frames...")

    writer.release()
    print(f"Done: {n_frames} frames, {n_kept_boxes} kept (on-court) detections total.")
    print(f"Saved to {a.out}")


if __name__ == "__main__":
    main()