"""Use a saved court_polygon.json to keep only people standing inside the marked region.

    from court_filter import CourtFilter
    cf = CourtFilter("court_polygon.json")
    ...
    for box in boxes_this_frame:          # box = (x1, y1, x2, y2)
        if cf.is_on_court(box):
            ...  # keep, feed into your size/presence scoring
"""
import json
import cv2
import numpy as np


class CourtFilter:
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.polygon = np.array(data["polygon"], dtype=np.int32)
        self.frame_size = tuple(data["frame_size"])  # (w, h) the polygon was drawn on

    def foot_point(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, y2)

    def is_on_court(self, box, frame_size=None):
        """frame_size: (w, h) of the frame `box` came from, if different from the
        frame the polygon was marked on (e.g. you resized frames for inference).
        Coordinates are scaled automatically so the polygon still lines up."""
        pt = self.foot_point(box)
        if frame_size is not None and tuple(frame_size) != self.frame_size:
            sx = self.frame_size[0] / frame_size[0]
            sy = self.frame_size[1] / frame_size[1]
            pt = (pt[0] * sx, pt[1] * sy)
        return cv2.pointPolygonTest(self.polygon, pt, False) >= 0
