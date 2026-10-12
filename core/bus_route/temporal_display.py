"""Causal ranking of observed display crops, never synthesis of missing strokes.

Relative stroke coverage is a quality gate, not proof of full-string readability.
The envelope is used only for scoring; OCR receives an actual source crop.
"""
from collections import deque

import cv2
import numpy as np


def display_crop_box(box, bounds):
    """Keep breathing room around observed text inside its physical bus.

    This does not fill missing digits. Both the detector box and expanded box
    must independently pass ownership/plate checks at the call site.
    """
    x,y,z,b = box
    bx,by,bz,bb = bounds
    height = b-y
    return (max(bx+2,round(x-.5*height)), max(by+2,round(y-.15*height)),
            min(bz-2,round(z+.5*height)), min(bb-2,round(b+.15*height)))


def crop_quality(rgb):
    gray = cv2.cvtColor(cv2.resize(rgb, (160, 48)), cv2.COLOR_RGB2GRAY)
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    dark_background = float(np.mean(border > threshold)) < .5
    if not dark_background:
        mask = 255-mask
    # A dark shadow at a tightly cropped PRINT border can invert the polarity.
    # Prefer the minority foreground; this is a pixel policy, not a LED label.
    if np.mean(mask > 0) > .65:
        mask = 255-mask
        dark_background = not dark_background
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    contrast = float(np.percentile(gray, 95)-np.percentile(gray, 5))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return {'mask': mask > 0, 'contrast': contrast, 'sharpness': sharpness,
            'dark_background': dark_background, 'ink_fraction': float(np.mean(mask > 0))}


class DisplayBuffer:
    def __init__(self, window_s=.4, min_span_s=.2, max_age_s=.25, coverage=.88):
        self.window_s = window_s
        self.min_span_s = min_span_s
        self.max_age_s = max_age_s
        self.coverage = coverage
        self.frames = deque()
        self.last_index = -1
        self.emitted = set()
        self.last_emitted_index = -1

    def add(self, frame, crop, box):
        index = frame['source_frame_index']
        now = frame['timestamp_s']
        if index <= self.last_index or (self.frames and now <= self.frames[-1]['timestamp_s']):
            raise ValueError('Duplicate or noncausal display observation')
        self.last_index = index
        if crop.size == 0:
            return
        item = {k: frame[k] for k in ('source_frame_index', 'source_pts', 'source_pts_s', 'timestamp_s')}
        item.update(crop=crop.copy(), box=tuple(box), quality=crop_quality(crop))
        self.frames.append(item)
        while self.frames and now-self.frames[0]['timestamp_s'] > self.window_s+1e-6:
            self.frames.popleft()
        self.emitted.intersection_update(item['source_frame_index'] for item in self.frames)

    def best(self, now, earliest_source_s=None):
        items = [item for item in self.frames if 0 <= now-item['timestamp_s'] <= self.window_s+1e-6]
        if len(items) < 3 or items[-1]['timestamp_s']-items[0]['timestamp_s'] < self.min_span_s-1e-6:
            return None, 'temporal_warmup'
        # Repeated observations suppress isolated glare/noise. No output pixels
        # are taken from this envelope, and no target string enters this score.
        masks = np.stack([item['quality']['mask'] for item in items])
        envelope = masks.sum(axis=0) >= max(2, int(np.ceil(len(items)*.15)))
        if envelope.sum() < 25:
            return None, 'insufficient_strokes'
        ranked = []
        for item in items:
            if now-item['timestamp_s'] > self.max_age_s+1e-6:
                continue
            if earliest_source_s is not None and item['timestamp_s'] < earliest_source_s:
                continue
            q = item['quality']
            observed = cv2.dilate(q['mask'].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            coverage = float((observed & envelope).sum()/envelope.sum())
            if coverage < self.coverage or q['contrast'] < 25 or q['sharpness'] < 15 or not .025 <= q['ink_fraction'] <= .65:
                continue
            score = coverage + .025*min(q['sharpness']/1000, 1) + .025*min(q['contrast']/150, 1)
            ranked.append((score, item['timestamp_s'], item, coverage))
        if not ranked:
            if earliest_source_s is not None and all(
                    item['timestamp_s'] < earliest_source_s for item in items):
                return None, 'no_fresh_evidence'
            return None, 'incomplete_or_low_quality'
        _, _, item, coverage = max(ranked, key=lambda x: (x[0], x[1]))
        if item['source_frame_index'] <= self.last_emitted_index:
            return None, 'already_observed_source_frame'
        self.emitted.add(item['source_frame_index'])
        self.last_emitted_index = item['source_frame_index']
        return {**item, 'relative_stroke_coverage': coverage}, 'observed_frame_selected'


def _display_tracking_points(gray, box):
    """Find the original display-ring corners without scanning the full frame."""
    x, y, z, b = map(int, box)
    height, width = gray.shape
    margin = max(12, round((b-y)*.6))
    left, top = max(0, x-margin), max(0, y-margin)
    right, bottom = min(width, z+margin), min(height, b+margin)
    if left >= right or top >= bottom:
        return None
    # Corner scores and nonmax suppression need neighboring pixels just outside
    # the mask. Keep that border while limiting the search to the display ring.
    pad = 8
    crop_left, crop_top = max(0, left-pad), max(0, top-pad)
    crop_right, crop_bottom = min(width, right+pad), min(height, bottom+pad)
    mask = np.zeros((crop_bottom-crop_top, crop_right-crop_left), np.uint8)
    mask[top-crop_top:bottom-crop_top, left-crop_left:right-crop_left] = 255
    # Lit characters change without physical motion. Track surrounding bus
    # structure, preserving the original full-frame mask's excluded rectangle.
    inner_left, inner_top = max(left, x), max(top, y)
    inner_right, inner_bottom = min(right, z), min(bottom, b)
    if inner_left < inner_right and inner_top < inner_bottom:
        mask[inner_top-crop_top:inner_bottom-crop_top,
             inner_left-crop_left:inner_right-crop_left] = 0
    points = cv2.goodFeaturesToTrack(
        gray[crop_top:crop_bottom, crop_left:crop_right], 50, .01, 3, mask=mask)
    if points is not None:
        points += np.array([[[crop_left, crop_top]]], np.float32)
    return points


def propagate_boxes(previous, current, boxes):
    """Local forward/backward optical flow; uncertain boxes are dropped.

    No matching across different bus objects; caller resets on selection epoch.
    """
    old = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    new = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    result = []
    for box in boxes:
        x, y, z, b = map(int, box)
        points = _display_tracking_points(old, box)
        if points is None or len(points) < 4:
            result.append(None)
            continue
        forward, status, _ = cv2.calcOpticalFlowPyrLK(old, new, points, None)
        if forward is None:
            result.append(None)
            continue
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(new, old, forward, None)
        if backward is None:
            result.append(None)
            continue
        good = (status[:, 0] > 0) & (back_status[:, 0] > 0) & (np.linalg.norm(backward-points, axis=2)[:, 0] < 1.5)
        if good.sum() < 4:
            result.append(None)
            continue
        affine, inliers = cv2.estimateAffinePartial2D(points[good], forward[good], method=cv2.RANSAC, ransacReprojThreshold=2)
        if affine is None or inliers.mean() < .6:
            result.append(None)
            continue
        scale = float(np.linalg.norm(affine[0, :2]))
        if not .9 <= scale <= 1.12:
            result.append(None)
            continue
        # Repeated floor/ceil of a rotated axis-aligned box expands it even on
        # stationary footage. Preserve its size under translation/scale.
        center = np.array([(x+z)/2,(y+b)/2,1]) @ affine.T
        half = np.array([z-x,b-y])*scale/2
        bounds = tuple(map(int, np.round([*(center-half),*(center+half)])))
        if bounds[0] <= 0 or bounds[1] <= 0 or bounds[2] >= new.shape[1]-1 or bounds[3] >= new.shape[0]-1:
            result.append(None)
        else:
            result.append(bounds)
    return result
