"""Pick the 2 real players out of everyone the court-polygon filter still lets
through (e.g. someone standing near the net, a spectator who leans onto court).

Idea: score each *track ID* over a rolling time window using two things,
computed only over frames where that track was on-court:
    - presence: fraction of the window the track was seen at all
    - size: median bounding-box height while on-court (bigger = closer to camera)
A track only becomes eligible once it's been seen enough (present_frac >= min_presence).
Among eligible tracks, keep the top N by median size. This is stable frame-to-frame
because it's judged over ~2-4 seconds of history, not a single frame.
"""
from collections import defaultdict, deque


class PlayerSelector:
    def __init__(self, window_frames=90, min_presence=0.5, n_players=2, always_fill=True):
        """
        window_frames : rolling window length, in frames (e.g. 90 @ 30fps = 3s)
        min_presence  : a track needs to be on-court in at least this fraction
                         of the window before it's "eligible" (trusted) to be picked
        n_players     : how many tracks to keep each frame (2 for doubles-side,
                         1 for singles-side, 4 if you ever track both sides at once)
        always_fill   : if True, when fewer than n_players tracks are eligible this
                         frame, fill the remaining slots with the next-biggest people
                         seen ON COURT this frame even if they haven't built up enough
                         history yet -- so you always get n_players picks when at least
                         that many people are on court, at the cost of occasionally
                         including someone not yet proven to be a real player (e.g. a
                         passerby's first second on court). Set False to only ever
                         return tracks that have passed the presence check.
        """
        self.window = window_frames
        self.min_presence = min_presence
        self.n_players = n_players
        self.always_fill = always_fill
        # per track_id: deque of heights, 0 = "not on court / not seen this frame"
        self._heights = defaultdict(lambda: deque(maxlen=window_frames))
        self._seen_ids_this_frame = set()
        self._this_frame_height = {}  # track_id -> height, this frame only (for fallback fill)

    def start_frame(self):
        """Call once at the start of processing each frame, before update()."""
        self._seen_ids_this_frame = set()
        self._this_frame_height = {}

    def update(self, track_id, box_height):
        """Call once per on-court detection this frame (i.e. AFTER the court-polygon
        filter already dropped off-court people). box_height = y2 - y1 in pixels."""
        self._heights[track_id].append(box_height)
        self._seen_ids_this_frame.add(track_id)
        self._this_frame_height[track_id] = box_height

    def end_frame(self):
        """Call once after all update() calls for this frame. Ages out tracks
        that were on-court before but not seen this frame, and returns the
        selected track IDs for THIS frame."""
        for tid in list(self._heights.keys()):
            if tid not in self._seen_ids_this_frame:
                self._heights[tid].append(0)  # counts as absent, drags presence down

        scored = []
        for tid, heights in self._heights.items():
            if len(heights) < min(self.window, 15):  # need at least half a second of history
                continue
            present_frac = sum(h > 0 for h in heights) / len(heights)
            if present_frac < self.min_presence:
                continue
            nonzero = sorted(h for h in heights if h > 0)
            median_h = nonzero[len(nonzero) // 2]
            scored.append((tid, median_h, present_frac))

        scored.sort(key=lambda x: -x[1])  # biggest (closest) first
        selected = {tid for tid, _, _ in scored[: self.n_players]}

        if self.always_fill and len(selected) < self.n_players:
            # Fill remaining slots from whoever is on-court THIS frame, ranked by
            # current height, skipping anyone already selected. These are flagged
            # as "unproven" in the returned info so callers can draw them differently
            # if desired (e.g. a dashed box instead of solid).
            candidates = [(tid, h) for tid, h in self._this_frame_height.items() if tid not in selected]
            candidates.sort(key=lambda x: -x[1])
            for tid, h in candidates:
                if len(selected) >= self.n_players:
                    break
                selected.add(tid)
                scored.append((tid, h, None))  # presence unknown/not yet proven

        return selected, scored  # scored returned too, useful for debugging/printing

    def prune(self, max_age_frames=None):
        """Optional: drop very old, fully-gone tracks so the dict doesn't grow forever
        over a long video. Call every N frames. Safe to skip for short clips."""
        max_age_frames = max_age_frames or self.window
        for tid in list(self._heights.keys()):
            heights = self._heights[tid]
            if len(heights) == heights.maxlen and all(h == 0 for h in heights):
                del self._heights[tid]