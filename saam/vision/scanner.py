"""Frame-to-frame scene tracking for scan mode.

SceneTracker is pure logic (no camera, no model), so it is unit-testable with
simulated detections. It turns noisy per-frame detections into sparse events:

    appeared    - a detection persisted for `persist_frames` consecutive frames
    disappeared - a confirmed object was missing for `absent_frames` frames
    moved       - a confirmed object stayed in a new zone / shifted by
                  `move_threshold` for `persist_frames` frames

`cooldown_s` limits how often the same (label, event) can be announced;
events held back by the cooldown are deferred, not dropped.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Tuple

from .detection import bbox_center


class _Track:
    __slots__ = ("id", "label", "center", "position", "confidence", "bbox",
                 "hits", "misses", "confirmed", "announced", "pending_appear",
                 "anchor_pos", "anchor_cx", "move_hits")

    def __init__(self, tid: int, det: dict):
        self.id = tid
        self.label = det["label"]
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.announced = False
        self.pending_appear = False
        self.anchor_pos = det["position"]
        self.anchor_cx = bbox_center(det["bbox"])[0]
        self.move_hits = 0
        self.refresh(det)

    def refresh(self, det: dict) -> None:
        self.center = bbox_center(det["bbox"])
        self.position = det["position"]
        self.confidence = det["confidence"]
        self.bbox = det["bbox"]


class SceneTracker:
    def __init__(self, persist_frames: int = 3, absent_frames: int = 4,
                 cooldown_s: float = 3.0, move_threshold: float = 0.25,
                 match_distance: float = 0.3):
        self.persist_frames = max(1, persist_frames)
        self.absent_frames = max(1, absent_frames)
        self.cooldown_s = cooldown_s
        self.move_threshold = move_threshold
        self.match_distance = match_distance
        self.reset()

    def reset(self) -> None:
        self._tracks: List[_Track] = []
        self._ids = itertools.count(1)
        self._last_emit: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    def update(self, detections: List[dict], now: Optional[float] = None) -> List[dict]:
        now = time.monotonic() if now is None else now
        events: List[dict] = []

        # 1. match detections to existing tracks (same label, nearest center)
        pairs = []
        for ti, tr in enumerate(self._tracks):
            for di, det in enumerate(detections):
                if det["label"] != tr.label:
                    continue
                dist = math.dist(tr.center, bbox_center(det["bbox"]))
                if dist <= self.match_distance:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._on_match(self._tracks[ti], detections[di], now, events)

        # 2. unmatched tracks
        survivors: List[_Track] = []
        for ti, tr in enumerate(self._tracks):
            if ti in used_t:
                survivors.append(tr)
                continue
            tr.misses += 1
            if not tr.confirmed:
                continue                       # persistence must be consecutive
            if tr.misses >= self.absent_frames:
                if tr.announced:
                    events.append(self._event("disappeared", tr, now))
                continue
            survivors.append(tr)
        self._tracks = survivors

        # 3. unmatched detections start new tracks
        for di, det in enumerate(detections):
            if di in used_d:
                continue
            tr = _Track(next(self._ids), det)
            self._tracks.append(tr)
            self._maybe_confirm(tr, now, events)
        return events

    # ------------------------------------------------------------------
    def _on_match(self, tr: _Track, det: dict, now: float, events: List[dict]) -> None:
        tr.refresh(det)
        tr.misses = 0
        tr.hits += 1
        if not tr.confirmed:
            self._maybe_confirm(tr, now, events)
            return
        if tr.pending_appear and self._cooldown_ok(tr.label, "appeared", now):
            self._emit_appeared(tr, now, events)
            return
        if not tr.announced:
            return
        shifted = (tr.position != tr.anchor_pos
                   or abs(tr.center[0] - tr.anchor_cx) >= self.move_threshold)
        tr.move_hits = tr.move_hits + 1 if shifted else 0
        if tr.move_hits >= self.persist_frames and self._cooldown_ok(tr.label, "moved", now):
            ev = self._event("moved", tr, now)
            ev["from_position"] = tr.anchor_pos
            ev["to_position"] = tr.position
            events.append(ev)
            self._last_emit[(tr.label, "moved")] = now
            tr.anchor_pos, tr.anchor_cx, tr.move_hits = tr.position, tr.center[0], 0

    def _maybe_confirm(self, tr: _Track, now: float, events: List[dict]) -> None:
        if tr.hits < self.persist_frames:
            return
        tr.confirmed = True
        tr.pending_appear = True
        if self._cooldown_ok(tr.label, "appeared", now):
            self._emit_appeared(tr, now, events)

    def _emit_appeared(self, tr: _Track, now: float, events: List[dict]) -> None:
        events.append(self._event("appeared", tr, now))
        self._last_emit[(tr.label, "appeared")] = now
        tr.pending_appear = False
        tr.announced = True
        tr.anchor_pos, tr.anchor_cx, tr.move_hits = tr.position, tr.center[0], 0

    def _cooldown_ok(self, label: str, kind: str, now: float) -> bool:
        last = self._last_emit.get((label, kind))
        return last is None or (now - last) >= self.cooldown_s

    @staticmethod
    def _event(kind: str, tr: _Track, now: float) -> dict:
        return {"event": kind, "label": tr.label, "position": tr.position,
                "confidence": tr.confidence, "track_id": tr.id, "timestamp": now}
