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
from player_selector import PlayerSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("polygon", help="court_polygon.json from mark_court.py")
    ap.add_argument("--out", default="annotated.avi")
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--draw-polygon", action="store_true", default=True,
                     help="also draw the court boundary on every frame")
    ap.add_argument("--n-players", type=int, default=2,
                     help="how many people to keep per frame after the size/presence filter")
    ap.add_argument("--window-seconds", type=float, default=3.0,
                     help="rolling window used to judge presence/size, in seconds")
    ap.add_argument("--min-presence", type=float, default=0.5,
                     help="a track must be on-court this fraction of the window to be eligible")
    ap.add_argument("--show-rejected", action="store_true",
                     help="also draw on-court-but-not-selected people in red, for debugging")
    ap.add_argument("--no-fallback-fill", action="store_true",
                     help="only ever select tracks that pass the presence check, even if "
                          "that means fewer than --n-players are selected some frames")
    a = ap.parse_args()

    cf = CourtFilter(a.polygon)
    model = YOLO(a.model)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    selector = PlayerSelector(
        window_frames=max(1, int(a.window_seconds * fps)),
        min_presence=a.min_presence,
        n_players=a.n_players,
        always_fill=not a.no_fallback_fill,
    )

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

        selector.start_frame()
        on_court = []  # (tid, box, kp) for people that pass the polygon filter this frame

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.tolist()
            xyxy = r.boxes.xyxy.tolist()
            kpts = r.keypoints.xy.tolist() if r.keypoints is not None else [None] * len(ids)

            for tid, box, kp in zip(ids, xyxy, kpts):
                x1, y1, x2, y2 = box
                if not cf.is_on_court((x1, y1, x2, y2), frame_size=frame_size):
                    continue  # stage 1: drop anyone outside the marked court region
                tid = int(tid)
                on_court.append((tid, box, kp))
                selector.update(tid, box_height=y2 - y1)

        selected_ids, scored = selector.end_frame()
        # presence=None marks a fallback-fill pick: selected to keep the count at
        # n_players, but not yet proven present/large enough on its own merits.
        unproven_ids = {tid for tid, _h, presence in scored if presence is None and tid in selected_ids}

        for tid, box, kp in on_court:
            x1, y1, x2, y2 = box
            is_selected = tid in selected_ids
            if not is_selected and not a.show_rejected:
                continue  # stage 2: drop on-court people who aren't one of the top-N by size/presence
            n_kept_boxes += 1 if is_selected else 0

            if not is_selected:
                color = (0, 0, 255)       # red: rejected (debug only, --show-rejected)
            elif tid in unproven_ids:
                color = (0, 255, 255)     # yellow: selected only as a fallback fill, not yet confident
            else:
                color = (0, 255, 0)       # green: confidently selected (passed presence/size check)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if is_selected and kp:
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