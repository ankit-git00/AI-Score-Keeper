"""Turn raw COCO keypoints into the per-player signals the serve detector needs.

All distances are normalized by that player's own body height (shoulder-to-ankle,
estimated per frame) so the same thresholds work regardless of how close/far the
camera is, or how zoomed in a given phone's shot is.
"""
from collections import deque

# COCO-17 keypoint indices (see earlier discussion)
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


def _valid(pt):
    x, y = pt
    return x > 0 and y > 0


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def body_scale(kp):
    """Rough body height in pixels this frame: shoulder-midpoint to ankle-midpoint.
    Falls back to hip-to-shoulder * 2.5 if ankles aren't visible (common when the
    lower body is cropped out or occluded). Returns None if not enough is visible."""
    ls, rs, lh, rh, la, ra = kp[L_SHOULDER], kp[R_SHOULDER], kp[L_HIP], kp[R_HIP], kp[L_ANKLE], kp[R_ANKLE]
    if not (_valid(ls) and _valid(rs)):
        return None
    shoulder = _mid(ls, rs)
    if _valid(la) and _valid(ra):
        ankle = _mid(la, ra)
        h = abs(ankle[1] - shoulder[1])
        if h > 5:
            return h
    if _valid(lh) and _valid(rh):
        hip = _mid(lh, rh)
        h = abs(hip[1] - shoulder[1]) * 2.5
        if h > 5:
            return h
    return None


def hip_point(kp):
    lh, rh = kp[L_HIP], kp[R_HIP]
    if _valid(lh) and _valid(rh):
        return _mid(lh, rh)
    return None


def wrist_points(kp):
    """Returns {'L': (x,y) or None, 'R': (x,y) or None}."""
    out = {}
    for side, idx in (("L", L_WRIST), ("R", R_WRIST)):
        out[side] = kp[idx] if _valid(kp[idx]) else None
    return out


class PlayerPoseTrack:
    """Keeps a short history of one player's pose so we can compute speeds
    (which need at least 2 frames) and smooth out single-frame jitter/noise."""

    def __init__(self, history_len=5):
        self.history = deque(maxlen=history_len)  # (t, kp, scale, hip, wrists)

    def update(self, t, kp):
        scale = body_scale(kp)
        hip = hip_point(kp)
        wrists = wrist_points(kp)
        self.history.append((t, kp, scale, hip, wrists))

    def features(self):
        """Returns a dict of current-frame features, or None if not enough data yet.

        body_speed       : hip movement per second, in body-heights/sec (0 = frozen)
        wrist_speed      : {'L':.., 'R':..} in body-heights/sec, None if wrist not visible
        wrist_rel_height : {'L':.., 'R':..} wrist y minus hip y, divided by body height.
                            Positive = wrist BELOW hip (serve-ready); negative = wrist raised.
        wrist_reach      : {'L':.., 'R':..} hip-to-wrist distance / body height (arm held out vs hanging)
        """
        if len(self.history) < 2:
            return None
        t0, kp0, s0, hip0, w0 = self.history[-2]
        t1, kp1, s1, hip1, w1 = self.history[-1]
        dt = t1 - t0
        if dt <= 0 or s1 is None or hip1 is None:
            return None
        scale = s1  # normalize by the most recent body-height estimate

        body_speed = None
        if hip0 is not None:
            dist = ((hip1[0] - hip0[0]) ** 2 + (hip1[1] - hip0[1]) ** 2) ** 0.5
            body_speed = dist / scale / dt

        wrist_speed = {}
        wrist_rel_height = {}
        wrist_reach = {}
        for side in ("L", "R"):
            p1 = w1.get(side)
            wrist_rel_height[side] = (p1[1] - hip1[1]) / scale if p1 else None
            # straight-line distance from hip to wrist, in body-heights: ~0 when the arm
            # hangs by the side, larger when the arm is held out from the body
            wrist_reach[side] = (((p1[0] - hip1[0]) ** 2 + (p1[1] - hip1[1]) ** 2) ** 0.5 / scale) if p1 else None
            p0 = w0.get(side)
            if p0 and p1:
                d = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
                wrist_speed[side] = d / scale / dt
            else:
                wrist_speed[side] = None

        return {
            "t": t1,
            "body_speed": body_speed,
            "wrist_speed": wrist_speed,
            "wrist_rel_height": wrist_rel_height,
            "wrist_reach": wrist_reach,
        }
