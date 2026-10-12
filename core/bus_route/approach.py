"""Target-independent selection in an explicitly configured camera/stop context.

No profile means abstention. A lane polygon is context, not a learned lane detector.
Epochs are deliberately not transferred across lost/overlapping/reassigned tracks.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / max(1, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


@dataclass(frozen=True)
class ApproachProfile:
    # Normalized bus bottom-center must be inside this lane/entry polygon.
    entry_polygon: tuple
    direction: tuple
    exit_polygon: tuple = ()
    min_history_s: float = .4
    min_growth: float = 1.15
    min_progress: float = .015
    max_gap_s: float = .3
    overlap_iou: float = .25

    def __post_init__(self):
        for polygon in (self.entry_polygon, self.exit_polygon):
            if polygon and (len(polygon) < 3 or any(len(p) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in polygon)):
                raise ValueError('Invalid normalized lane polygon')
        if not self.entry_polygon or len(self.direction) != 2 or not all(math.isfinite(v) for v in self.direction) or math.hypot(*self.direction) == 0:
            raise ValueError('Lane context and nonzero direction required')
        if not (0 < self.min_history_s <= 2 and self.min_growth >= 1 and self.min_progress >= 0 and 0 < self.max_gap_s <= 1 and 0 < self.overlap_iou < 1):
            raise ValueError('Invalid approach thresholds')


class ApproachSelector:
    def __init__(self, profile=None):
        self.profile = profile
        self.history = {}
        self.focus = None
        self.epoch = 0
        self.last = None
        self.box = None
        self.retired = set()
        self.reason = 'unconfigured_camera_context'
        self.candidate_diagnostics = []

    def reset(self, reason):
        self.focus = None
        self.box = None
        self.history.clear()
        self.epoch += 1
        self.reason = reason
        self.candidate_diagnostics = []

    @staticmethod
    def geometry(box, shape):
        h, w = shape[:2]
        x, y, z, b = box
        return np.array([(x+z)/2/w, b/h]), (z-x)*(b-y)/(w*h)

    @staticmethod
    def inside(point, polygon):
        return bool(polygon) and cv2.pointPolygonTest(np.array(polygon, np.float32), tuple(map(float, point)), False) >= 0

    def update(self, timestamp, buses, shape, camera_stable=True, camera_pan=False):
        if self.last is not None and timestamp <= self.last:
            raise ValueError('Selection timestamps must strictly increase')
        previous_time = self.last
        self.last = timestamp
        self.candidate_diagnostics = []
        p = self.profile
        if p is None:
            self.candidate_diagnostics = [{'track_id': b['track_id'],
                                           'reason': 'unconfigured_camera_context'} for b in buses]
            return None
        if camera_pan and self.focus is None:
            # A moving camera cannot establish a new approach from a profile
            # tied to the previous view. History must restart when stable.
            self.reset('camera_pan_without_selected_focus')
            self.candidate_diagnostics = [{'track_id': b['track_id'],
                                           'reason': self.reason} for b in buses]
            return None
        if not camera_stable and not camera_pan:
            self.reset('camera_motion_or_unverified_context')
            self.candidate_diagnostics = [{'track_id': b['track_id'],
                                           'reason': self.reason} for b in buses]
            return None
        current = {b['track_id']: b for b in buses if b['track_id'] is not None}
        if len(current) != sum(b['track_id'] is not None for b in buses):
            self.reset('duplicate_track_id')
            self.candidate_diagnostics = [{'track_id': b['track_id'],
                                           'reason': self.reason} for b in buses]
            return None
        if self.focus is not None:
            bus = current.get(self.focus)
            reason = None
            if bus is None or timestamp-previous_time > p.max_gap_s+1e-6:
                reason = 'track_lost'
            elif iou(self.box, bus['bus_box']) < .1:
                reason = 'identity_discontinuity'
            elif any(other['track_id'] != self.focus and iou(bus['bus_box'], other['bus_box']) >= p.overlap_iou for other in buses):
                reason = 'overlapping_identity'
            elif self.inside(self.geometry(bus['bus_box'], shape)[0], p.exit_polygon):
                self.retired.add(self.focus)
                reason = 'passed_stop'
            if reason:
                lost_focus = self.focus
                self.reset(reason)
                self.candidate_diagnostics = [{'track_id': lost_focus,
                                               'reason': reason}]
                return None
        for key in list(self.history):
            if key not in current:
                del self.history[key]
        for key, bus in current.items():
            history = self.history.setdefault(key, [])
            if history and (timestamp-history[-1][0] > p.max_gap_s+1e-6 or iou(history[-1][1], bus['bus_box']) < .1):
                history.clear()
            history.append((timestamp, bus['bus_box']))
            self.history[key] = [item for item in history if timestamp-item[0] <= 1.2]
        if self.focus is None:
            candidates = []
            direction = np.array(p.direction) / math.hypot(*p.direction)
            for key, bus in current.items():
                if key in self.retired:
                    self.candidate_diagnostics.append({'track_id': key, 'reason': 'retired_track'})
                    continue
                history = self.history[key]
                point, area = self.geometry(bus['bus_box'], shape)
                if len(history) < 3:
                    self.candidate_diagnostics.append({'track_id': key, 'reason': 'insufficient_history'})
                    continue
                if timestamp-history[0][0] < p.min_history_s-1e-6:
                    self.candidate_diagnostics.append({'track_id': key, 'reason': 'insufficient_history_span'})
                    continue
                if not self.inside(point, p.entry_polygon):
                    self.candidate_diagnostics.append({'track_id': key, 'reason': 'outside_entry_polygon'})
                    continue
                old_point, old_area = self.geometry(history[0][1], shape)
                if not self.inside(old_point, p.entry_polygon):
                    self.candidate_diagnostics.append({'track_id': key, 'reason': 'history_started_outside_entry_polygon'})
                    continue
                if area < old_area*p.min_growth:
                    self.candidate_diagnostics.append({'track_id': key, 'reason': 'insufficient_growth'})
                    continue
                if np.dot(point-old_point, direction) < p.min_progress:
                    self.candidate_diagnostics.append({'track_id': key, 'reason': 'insufficient_progress'})
                    continue
                if any(other['track_id'] != key and iou(bus['bus_box'], other['bus_box']) >= p.overlap_iou for other in buses):
                    self.candidate_diagnostics.append({'track_id': key, 'reason': 'overlapping_identity'})
                    continue
                self.candidate_diagnostics.append({'track_id': key, 'reason': 'candidate'})
                candidates.append(key)
            if len(candidates) != 1:
                self.reason = 'ambiguous_approach' if candidates else 'awaiting_approach_evidence'
                return None
            self.focus = candidates[0]
            self.epoch += 1
        self.box = current[self.focus]['bus_box']
        self.reason = 'selected_approach_pan_hold' if camera_pan else 'selected_approach'
        if camera_pan or not self.candidate_diagnostics:
            self.candidate_diagnostics = [{'track_id': key,
                                           'reason': self.reason if key == self.focus else 'another_focus_selected'}
                                          for key in current]
        return current[self.focus]
