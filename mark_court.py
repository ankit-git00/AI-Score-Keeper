"""Click out a court-region polygon on a frame from your video.

Usage:
    python mark_court.py your_video.mp4 --time 5.0

Controls:
    Left click   : add a point (click around the region boundary, in order --
                   either all the way clockwise or all the way counter-clockwise,
                   don't zig-zag back and forth)
    u            : undo last point
    s            : save polygon to court_polygon.json and print it
    r            : reset (clear all points)
    q / Esc      : quit without saving

Notes:
    - The polygon does NOT need to follow painted court lines. If you can't see
      the far lines (camera on the ground), just click around wherever players
      actually stand/walk on your side, using whatever landmarks are visible
      (edge of the green floor, a wall marking, the net post, etc).
    - Doesn't need to be a rectangle -- any simple (non-self-crossing) shape works.
    - Click at least 3 points. 4-6 is typical.
"""
import argparse
import json
import cv2
import numpy as np

points = []
img = None
win = "mark court (u=undo, s=save, r=reset, q=quit)"


def redraw():
    disp = img.copy()
    for i, p in enumerate(points):
        cv2.circle(disp, p, 5, (0, 0, 255), -1)
        cv2.putText(disp, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if len(points) >= 2:
        cv2.polylines(disp, [np.array(points, np.int32)], isClosed=len(points) >= 3,
                      color=(0, 255, 0), thickness=2)
    cv2.imshow(win, disp)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()


def main():
    global img
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--time", type=float, default=5.0, help="seconds into the video to grab the frame from")
    ap.add_argument("--out", default="court_polygon.json")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.time * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame at that timestamp -- try a different --time.")

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    print(__doc__)

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            print("Quit without saving.")
            break
        elif k == ord("u") and points:
            points.pop()
            redraw()
        elif k == ord("r"):
            points.clear()
            redraw()
        elif k == ord("s"):
            if len(points) < 3:
                print("Need at least 3 points before saving.")
                continue
            data = {"video": a.video, "frame_time_s": a.time,
                    "frame_size": [img.shape[1], img.shape[0]], "polygon": points}
            with open(a.out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(points)} points to {a.out}:")
            print(points)
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
