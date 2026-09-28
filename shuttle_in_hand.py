"""Is a selected player holding a (yellow-green) shuttle right now?

How it decides, per frame:
  1. Find yellow-green blobs in the whole frame (HSV colour threshold).
  2. Learn which blobs are static background (e.g. a yellow bin lid on a bench):
     a blob that keeps appearing in the same place while it is NOT on a selected
     player's body gets ignored from then on.
  3. Assign each remaining blob to the nearest wrist (of a selected player) within
     a radius scaled to that player's body size. Too-small or too-big blobs are dropped.
  4. A wrist "holds" the shuttle while a blob is assigned to it. Short dropouts
     (motion blur, fingers covering it) are bridged, and each hold becomes an
     interval with a start, an end and a duration.

Colour limits are for the yellow-green plastic shuttle seen in the test videos and are
a starting point tuned on ONE frame. Different phones/lighting will shift them.
"""
from collections import defaultdict

import cv2
import numpy as np

from pose_features import body_scale, wrist_points


class ShuttleColor:
    def __init__(self, h=(22, 45), s=(90, 255), v=(140, 255), min_area=20):
        self.lo = (h[0], s[0], v[0])
        self.hi = (h[1], s[1], v[1])
        self.min_area = min_area
        self._kernel = np.ones((2, 2), np.uint8)

    def blobs(self, frame_bgr):
        """List of (area_px, cx, cy) for yellow-green blobs at least min_area big."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.lo, self.hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
        n, _lab, st, cent = cv2.connectedComponentsWithStats(m)
        out = []
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] >= self.min_area:
                out.append((int(st[k, cv2.CC_STAT_AREA]), float(cent[k, 0]), float(cent[k, 1])))
        return out


class ShuttleInHand:
    def __init__(self, fps, color=None, radius_body=0.15, min_radius_px=18,
                 max_area_body=0.1, bg_seconds=2.0, cell=16,
                 hold_gap_s=0.4, min_interval_s=0.3, box_pad=0.15):
        """
        radius_body    : how close a blob must be to a wrist, as a fraction of body height
        min_radius_px  : ...but never tighter than this many pixels
        max_area_body  : blobs bigger than (this * body height)^2 pixels are not a held shuttle
        bg_seconds     : a blob seen this long in one spot, off any selected player, is background
        hold_gap_s     : a hold survives the blob vanishing for up to this long
        min_interval_s : holds shorter than this are treated as noise and not reported
        box_pad        : selected players' boxes are grown by this fraction before deciding a
                         blob is "not on a player" (so their own blobs never become background)
        """
        self.fps = fps
        self.color = color or ShuttleColor()
        self.radius_body = radius_body
        self.min_radius_px = min_radius_px
        self.max_area_body = max_area_body
        self.bg_frames = max(1, int(bg_seconds * fps))
        self.cell = cell
        self.hold_gap_s = hold_gap_s
        self.min_interval_s = min_interval_s
        self.box_pad = box_pad

        self._bg = defaultdict(int)     # (cell_x, cell_y) -> frames a blob was seen there off-player
        self._since = {}                # (tid, side) -> t the current hold began
        self._last = {}                 # (tid, side) -> last t the shuttle was seen at that wrist
        self.intervals = []             # finished holds: dicts with track_id, side, start, end, duration

    # ---- background learning -------------------------------------------------
    def _cell(self, cx, cy):
        return int(cx // self.cell), int(cy // self.cell)

    def _is_background(self, cx, cy):
        c0, c1 = self._cell(cx, cy)
        total = sum(self._bg.get((c0 + dx, c1 + dy), 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        return total >= self.bg_frames

    def _on_player(self, cx, cy, boxes):
        for (x1, y1, x2, y2) in boxes:
            pw, ph = (x2 - x1) * self.box_pad, (y2 - y1) * self.box_pad
            if x1 - pw <= cx <= x2 + pw and y1 - ph <= cy <= y2 + ph:
                return True
        return False

    # ---- main per-frame call -------------------------------------------------
    def update(self, t, frame_bgr, players):
        """players: {track_id: {"kp": 17 (x, y) pairs, "box": (x1, y1, x2, y2)}} for the SELECTED
        players this frame, in the same pixel coordinates as frame_bgr.

        Returns {"assigned": {(tid, side): (area, cx, cy)},   # wrists holding a shuttle now
                 "background": [(area, cx, cy)],             # blobs ignored as static background
                 "unassigned": [(area, cx, cy)],             # blobs not near any tracked wrist
                 "holds": {(tid, side): seconds}}            # current hold length per wrist
        """
        blobs = self.color.blobs(frame_bgr)
        boxes = [p["box"] for p in players.values()]

        # learn background from blobs that are not on a selected player
        for _a, cx, cy in blobs:
            if not self._on_player(cx, cy, boxes):
                self._bg[self._cell(cx, cy)] += 1

        background, candidates = [], []
        for b in blobs:
            (background if self._is_background(b[1], b[2]) else candidates).append(b)

        # wrists we can locate, each with its own search radius and size limits
        wrists = []
        for tid, p in players.items():
            kp = p["kp"]
            scale = body_scale(kp)
            if scale is None:
                x1, y1, x2, y2 = p["box"]
                scale = 0.8 * (y2 - y1)
            for side, pt in wrist_points(kp).items():
                if pt is not None:
                    wrists.append((tid, side, pt, max(self.min_radius_px, self.radius_body * scale),
                                   (self.max_area_body * scale) ** 2))

        # every blob goes to its nearest wrist (if within that wrist's radius and size limit);
        # each wrist keeps the biggest blob it was given
        best = {}
        claimed = set()
        for b in candidates:
            area, cx, cy = b
            near = None
            for tid, side, pt, radius, max_area in wrists:
                d = ((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2) ** 0.5
                if d <= radius and area <= max_area and (near is None or d < near[0]):
                    near = (d, (tid, side))
            if near is not None:
                key = near[1]
                claimed.add(b)
                if key not in best or area > best[key][0]:
                    best[key] = b
        unassigned = [b for b in candidates if b not in claimed]

        self._update_holds(t, set(best.keys()), players)
        return {"assigned": best, "background": background, "unassigned": unassigned,
                "holds": {k: self.current_hold_s(*k, t) for k in best}}

    # ---- hold bookkeeping ----------------------------------------------------
    def _update_holds(self, t, present_keys, players):
        keys = set(self._since.keys()) | present_keys
        for tid in players:
            keys |= {(tid, "L"), (tid, "R")}
        for key in keys:
            if key in present_keys:
                if self._since.get(key) is None:
                    self._since[key] = t
                self._last[key] = t
            elif self._since.get(key) is not None and t - self._last[key] > self.hold_gap_s:
                self._close(key)

    def _close(self, key):
        start, end = self._since[key], self._last[key]
        if end - start >= self.min_interval_s:
            self.intervals.append({"track_id": key[0], "side": key[1],
                                   "start": round(start, 3), "end": round(end, 3),
                                   "duration": round(end - start, 3)})
        self._since[key] = None

    def current_hold_s(self, tid, side, t):
        key = (tid, side)
        since = self._since.get(key)
        if since is None or t - self._last.get(key, since) > self.hold_gap_s:
            return 0.0
        return self._last[key] - since

    def finish(self):
        """Call once at the end of the video to close any hold still open."""
        for key in [k for k, v in self._since.items() if v is not None]:
            self._close(key)
        return self.intervals
