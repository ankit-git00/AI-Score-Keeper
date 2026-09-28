"""Detect when the selected players are holding a yellow-green shuttle.

Runs: YOLO-pose tracking -> court polygon -> top-N player selection -> shuttle-in-hand.
Writes:
    <out>.avi        overlay video (watch this to judge the detector)
    <out>_holds.json list of hold intervals {track_id, side, start, end, duration}

Overlay legend:
    green box              selected player
    small blue dot         a wrist with no shuttle
    big red ring + HOLD    wrist judged to be holding a shuttle (with seconds so far)
    white ring             yellow-green blob not near any selected wrist (ignored)
    grey ring              blob ignored as static background

Usage:
    python detect_holds.py cleanVideo2.mp4 court_polygon.json --out holds_v2
Requires: pip install ultralytics opencv-python
"""
import argparse
import json

import cv2
from ultralytics import YOLO

from court_filter import CourtFilter
from player_selector import PlayerSelector
from shuttle_in_hand import ShuttleColor, ShuttleInHand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("polygon")
    ap.add_argument("--out", default="holds")
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-players", type=int, default=2)
    ap.add_argument("--window-seconds", type=float, default=3.0)
    ap.add_argument("--min-presence", type=float, default=0.5)
    # shuttle colour (OpenCV HSV: H 0-179, S/V 0-255) and size limits
    ap.add_argument("--hue", type=int, nargs=2, default=(22, 45))
    ap.add_argument("--min-sat", type=int, default=90)
    ap.add_argument("--min-val", type=int, default=140)
    ap.add_argument("--min-area", type=int, default=20, help="smallest blob (pixels) that can be a shuttle")
    ap.add_argument("--radius-body", type=float, default=0.15, help="max blob-to-wrist distance / body height")
    ap.add_argument("--hold-gap-s", type=float, default=0.4)
    ap.add_argument("--min-interval-s", type=float, default=0.3)
    a = ap.parse_args()

    cf = CourtFilter(a.polygon)
    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    selector = PlayerSelector(window_frames=max(1, int(a.window_seconds * fps)),
                              min_presence=a.min_presence, n_players=a.n_players)
    color = ShuttleColor(h=tuple(a.hue), s=(a.min_sat, 255), v=(a.min_val, 255), min_area=a.min_area)
    det = ShuttleInHand(fps=fps, color=color, radius_body=a.radius_body,
                        hold_gap_s=a.hold_gap_s, min_interval_s=a.min_interval_s)
    writer = cv2.VideoWriter(f"{a.out}.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

    n = 0
    for r in model.track(a.video, persist=True, device=a.device, stream=True, verbose=False):
        t = n / fps
        frame = r.orig_img.copy()
        fsize = (r.orig_shape[1], r.orig_shape[0])

        selector.start_frame()
        on_court = {}
        if r.boxes is not None and r.boxes.id is not None:
            kpts = r.keypoints.xy.tolist() if r.keypoints is not None else [None] * len(r.boxes.id)
            for tid, box, kp in zip(r.boxes.id.tolist(), r.boxes.xyxy.tolist(), kpts):
                if kp is None or not cf.is_on_court(tuple(box), frame_size=fsize):
                    continue
                tid = int(tid)
                selector.update(tid, box[3] - box[1])
                on_court[tid] = {"kp": [tuple(p) for p in kp], "box": tuple(box)}
        selected, _ = selector.end_frame()
        players = {tid: p for tid, p in on_court.items() if tid in selected}

        res = det.update(t, r.orig_img, players)

        # ---- overlay ----
        for tid, p in players.items():
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            for side, idx in (("L", 9), ("R", 10)):
                x, y = p["kp"][idx]
                if x > 0 and y > 0 and (tid, side) not in res["assigned"]:
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 120, 0), -1)
        for (tid, side), (area, cx, cy) in res["assigned"].items():
            cv2.circle(frame, (int(cx), int(cy)), 16, (0, 0, 255), 3)
            cv2.putText(frame, f"HOLD {side} {res['holds'][(tid, side)]:.1f}s", (int(cx) + 18, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for _a, cx, cy in res["unassigned"]:
            cv2.circle(frame, (int(cx), int(cy)), 10, (255, 255, 255), 1)
        for _a, cx, cy in res["background"]:
            cv2.circle(frame, (int(cx), int(cy)), 10, (140, 140, 140), 1)
        cv2.putText(frame, f"t={t:6.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)

        n += 1
        if n % 300 == 0:
            print(f"...{n} frames ({t:.0f}s)")

    writer.release()
    intervals = det.finish()
    with open(f"{a.out}_holds.json", "w") as f:
        json.dump(intervals, f, indent=2)
    print(f"{n} frames. {len(intervals)} hold intervals:")
    for i in intervals:
        print(f"  ID {i['track_id']} {i['side']}: {i['start']:7.2f}s -> {i['end']:7.2f}s  ({i['duration']:.1f}s)")
    print(f"Wrote {a.out}.avi and {a.out}_holds.json")


if __name__ == "__main__":
    main()
