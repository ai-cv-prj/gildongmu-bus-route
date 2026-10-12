"""Per-bus display buffers with OCR only for the selected bus.

This opt-in path requires a camera-context profile and fixed route weights.
It does not turn an unknown lane, a detector box, or a high OCR score into a
human-verified service target/readability label.
"""
from dataclasses import asdict
import time

import cv2
import numpy as np
from PIL import Image

from .approach import ApproachSelector, iou
from .route_evidence import crop_rejection, text_rejection, context_boxes, context_rejection
from .target import TargetMatcher
from .temporal_display import DisplayBuffer, propagate_boxes, display_crop_box


def camera_stable(previous, current, buses, diagnostics=None):
    """Measure background motion; report why a frame cannot certify stability.

    The optional diagnostics record is observational. In particular, too few
    background tracks and measured motion remain different failure modes.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    if previous is None:
        diagnostics['status'] = 'first_frame'
        return True
    height, width = current.shape[:2]
    size = (320, max(32, round(height*320/width)))
    old = cv2.cvtColor(cv2.resize(previous, size), cv2.COLOR_RGB2GRAY)
    new = cv2.cvtColor(cv2.resize(current, size), cv2.COLOR_RGB2GRAY)
    mask = np.full(old.shape, 255, np.uint8)
    for bus in buses:
        x,y,z,b = bus['bus_box']
        mask[max(0,int(y*size[1]/height)):min(size[1],int(b*size[1]/height)+1),
             max(0,int(x*size[0]/width)):min(size[0],int(z*size[0]/width)+1)] = 0
    points = cv2.goodFeaturesToTrack(old, 100, .02, 5, mask=mask)
    diagnostics['background_fraction'] = float(np.count_nonzero(mask) / mask.size)
    diagnostics['point_count'] = 0 if points is None else len(points)
    if points is None or len(points) < 10:
        diagnostics['status'] = 'insufficient_points'
        return False
    moved, status, _ = cv2.calcOpticalFlowPyrLK(old, new, points, None)
    diagnostics['flow_count'] = 0 if moved is None or status is None else int(status.sum())
    if moved is None or status is None or status.sum() < 10:
        diagnostics['status'] = 'insufficient_flow'
        return False
    delta = (moved-points)[status[:,0] > 0, 0]
    median = np.median(delta, axis=0)
    residual = np.linalg.norm(delta-median, axis=1)
    motion = float(np.linalg.norm(median) / 320)
    diagnostics.update(median_delta_px=[float(v) for v in median],
                       median_residual_px=float(np.median(residual)),
                       inlier_fraction_3px=float(np.mean(residual <= 3)),
                       normalized_motion=motion,
                       status='stable' if motion < .018 else 'measured_motion')
    return motion < .018


class ApproachPipeline:
    def __init__(self, detector, recognizer, target, profile=None, detection_interval=.2,
                 temporal=True, camera_check=True, preselection_collection=False,
                 camera_pan_hold=True, realtime_freshness=True):
        self.detector = detector
        self.recognizer = recognizer
        self.target = target
        self.selector = ApproachSelector(profile)
        self.detection_interval = detection_interval
        self.temporal = temporal
        self.camera_check = camera_check
        self.preselection_collection = preselection_collection
        self.camera_pan_hold = camera_pan_hold
        self.realtime_freshness = realtime_freshness
        self.matcher = TargetMatcher(target, single_score=.98)
        self.last_detection = -1e9
        self.previous = None
        self.detection_image = None
        self.buses = []
        self.displays = {}
        self.next_display = 0
        self.epoch = 0
        self.sources = {}
        self.observed_sources = set()
        self.track_boxes = {}
        self.raw_frames = {}
        self.fallback_attempted = set()
        self.context_stable = False
        self.last_camera_delta = None
        self.pan_started_at = None
        self.camera_diagnostics = {}
        self.detector.reset()

    def _camera_state(self, now, rgb, buses):
        """Allow only a short, coherent pan to retain an existing bus focus.

        A pan cannot create a new focus. Track continuity, overlap, and exit
        checks still run in ApproachSelector before any display/OCR work.
        """
        detail = {}
        stable = True if not self.camera_check else camera_stable(
            self.detection_image, rgb, self.buses+buses, diagnostics=detail)
        if not self.camera_check:
            detail['status'] = 'check_disabled'
        delta = detail.get('median_delta_px')
        previous = self.last_camera_delta
        pan = bool(not stable and self.camera_pan_hold and self.selector.focus is not None
                   and detail.get('status') == 'measured_motion' and delta is not None
                   and previous is not None
                   and abs(delta[0]) <= 10 and abs(delta[1]) <= 3
                   and detail['median_residual_px'] <= 2
                   and detail['inlier_fraction_3px'] >= .5
                   and delta[0]*previous[0] > 0
                   and abs(delta[0]-previous[0]) <= 4
                   and abs(delta[1]-previous[1]) <= 2.5
                   and (self.pan_started_at is None or now-self.pan_started_at <= .4+1e-6))
        if pan and self.pan_started_at is None:
            self.pan_started_at = now
        if not pan:
            self.pan_started_at = None
        self.last_camera_delta = delta
        detail['pan_hold'] = pan
        detail['pan_age_s'] = now-self.pan_started_at if pan else None
        self.camera_diagnostics = detail
        return stable, pan

    def _earliest_live_source(self, frame):
        """Keep OCR on a real source frame whose 300 ms deadline is viable."""
        deadline = frame.get('wall_deadline_monotonic') if self.realtime_freshness else None
        if deadline is None:
            return None
        # Reserve a small part of the remaining budget for OCR and the final
        # context/decision checks. The outer replay still enforces 300 ms.
        return frame['timestamp_s'] - (deadline-time.perf_counter()-.025)

    def _clear(self, keep_track=None):
        self.displays = {key: value for key, value in self.displays.items()
                         if keep_track is not None and value.get('track_id') == keep_track}
        self.sources = {key: value for key, value in self.sources.items()
                        if keep_track is not None and isinstance(key, tuple) and key[0] == keep_track}
        self.observed_sources = {key for key in self.observed_sources
                                 if keep_track is not None and isinstance(key, tuple) and key[0] == keep_track}
        self.track_boxes = {key: value for key, value in self.track_boxes.items() if key == keep_track}
        self.raw_frames = {key: value for key, value in self.raw_frames.items() if key == keep_track}
        self.fallback_attempted = {key for key in self.fallback_attempted if key == keep_track}
        previous = self.matcher.tracks.get(keep_track)
        self.matcher = TargetMatcher(self.target, single_score=.98)
        if previous is not None:
            self.matcher.tracks[keep_track] = previous
        self.epoch = self.selector.epoch

    def _drop_track(self, track_id):
        self.displays = {key: value for key, value in self.displays.items()
                         if value['track_id'] != track_id}
        self.sources = {key: value for key, value in self.sources.items() if key[0] != track_id}
        self.observed_sources = {key for key in self.observed_sources if key[0] != track_id}
        self.track_boxes.pop(track_id, None)
        self.raw_frames.pop(track_id, None)
        self.fallback_attempted.discard(track_id)
        self.matcher.tracks.pop(track_id, None)

    def _allowed(self, box, bus, rgb, buses=None):
        x,y,z,b = box
        if z-x < 10 or b-y < 6 or x <= 0 or y <= 0 or z >= rgb.shape[1]-1 or b >= rgb.shape[0]-1:
            return False
        bx,by,bz,bb = bus['bus_box']
        if not (bx < x < z < bz and by < y < b < bb):
            return False
        # Require the entire candidate to be unambiguous, not just its center.
        for other in self.buses if buses is None else buses:
            if other['track_id'] == bus['track_id']:
                continue
            a,c,d,e = other['bus_box']
            intersection = max(0,min(z,d)-max(x,a))*max(0,min(b,e)-max(y,c))
            if intersection / ((z-x)*(b-y)) > .1:
                return False
        return not crop_rejection(rgb, box, bus['bus_box'], bus['bus_box'])

    @staticmethod
    def _project_box(box, source_bus_box, destination_bus_box):
        sx, sy, sz, sb = source_bus_box
        dx, dy, dz, db = destination_bus_box
        if sz <= sx or sb <= sy or dz <= dx or db <= dy:
            return None
        return tuple(map(round, (dx+(box[0]-sx)*(dz-dx)/(sz-sx),
                                 dy+(box[1]-sy)*(db-dy)/(sb-sy),
                                 dx+(box[2]-sx)*(dz-dx)/(sz-sx),
                                 dy+(box[3]-sy)*(db-dy)/(sb-sy))))

    def _remember_preselection_frame(self, track_id, frame, bus, detection_due):
        ring = self.raw_frames.setdefault(track_id, [])
        ring.append({'frame': frame, 'bus_box': tuple(bus['bus_box']),
                     'buses': tuple(self.buses), 'detected_at': self.last_detection,
                     'detection_due': detection_due})
        now = frame['timestamp_s']
        ring[:] = [item for item in ring if now-item['frame']['timestamp_s'] <= .4+1e-6]
        if len(ring) > 5:
            latest_detection = next((item for item in reversed(ring) if item['detection_due']), None)
            near_span = min(ring, key=lambda item: abs(item['frame']['timestamp_s']-(now-.22)))
            chosen = [*ring[-3:], near_span]
            if latest_detection is not None:
                chosen.append(latest_detection)
            seen = {item['frame']['source_frame_index'] for item in chosen}
            if len(seen) < 5:
                chosen.append(ring[0])
            wanted = {item['frame']['source_frame_index'] for item in chosen}
            ring[:] = [item for item in ring if item['frame']['source_frame_index'] in wanted]

    @staticmethod
    def _historical_bus_box(snapshot, now, current_bus_box):
        old = snapshot['bus_box']
        seen_at = snapshot['detected_at']
        at = snapshot['frame']['timestamp_s']
        if now <= seen_at or at <= seen_at:
            return old
        fraction = min(1., max(0., (at-seen_at)/(now-seen_at)))
        return tuple(round(a+(b-a)*fraction) for a,b in zip(old,current_bus_box))

    def _preselection_buses(self, now, selected, detection_due):
        """Buffer only continuous, separately identified buses in the entry area."""
        if not self.context_stable:
            return {}
        profile = self.selector.profile
        if self.selector.focus is not None:
            candidates = [selected] if selected is not None else []
        else:
            candidates = [bus for bus in self.buses
                          if self.selector.inside(self.selector.geometry(bus['bus_box'],
                                                                         self.detection_image.shape)[0],
                                                  profile.entry_polygon)]
        active = {}
        for bus in candidates:
            track_id = bus['track_id']
            if track_id is None:
                continue
            if any(other is not bus and iou(bus['bus_box'], other['bus_box']) >= profile.overlap_iou
                   for other in self.buses):
                if detection_due:
                    self._drop_track(track_id)
                continue
            previous = self.track_boxes.get(track_id)
            if detection_due and previous is not None:
                seen_at, old_box = previous
                if now-seen_at > profile.max_gap_s+1e-6 or iou(old_box, bus['bus_box']) < .1:
                    self._drop_track(track_id)
            if detection_due:
                self.track_boxes[track_id] = (now, bus['bus_box'])
            if track_id in self.track_boxes:
                active[track_id] = bus
        if detection_due:
            for track_id in set(self.track_boxes)-set(active):
                self._drop_track(track_id)
        return active

    def _seed_selected_displays(self, now, selected):
        track_id = selected['track_id']
        for key, display in self.displays.items():
            if 'retrospective_buffer' in display:
                continue
            retro = DisplayBuffer()
            for snapshot in self.raw_frames.get(track_id, []):
                source = snapshot['frame']
                if not 0 < now-source['timestamp_s'] <= .4+1e-6:
                    continue
                old_bus_box = self._historical_bus_box(snapshot, now, selected['bus_box'])
                old_bus = {'track_id': track_id, 'bus_box': old_bus_box}
                old_box = self._project_box(display['box'], selected['bus_box'], old_bus_box)
                if old_box is None or not self._allowed(old_box, old_bus, source['rgb'], snapshot['buses']):
                    continue
                x,y,z,b = old_box
                retro.add(source, source['rgb'][y:b,x:z], old_box)
                self.sources[(track_id, source['source_frame_index'])] = source
            display['retrospective_buffer'] = retro

    def _accepted_displays(self, boxes, bus, rgb, buses=None):
        accepted = []
        for detector_box in boxes:
            if not self._allowed(detector_box, bus, rgb, buses):
                continue
            box = display_crop_box(detector_box, bus['bus_box'])
            if self._allowed(box, bus, rgb, buses):
                accepted.append((tuple(detector_box), tuple(box)))
        return accepted

    def _fallback_display_boxes(self, frame, selected):
        rgb, now = frame['rgb'], frame['timestamp_s']
        track_id = selected['track_id']
        if track_id in self.fallback_attempted:
            return [], None
        ring = [item for item in self.raw_frames.get(track_id, [])
                if 0 < now-item['frame']['timestamp_s'] <= .25+1e-6]
        candidates = [item for item in ring if item['detection_due']] or ring
        if not candidates:
            return [], 'retrospective_route_scan_no_recent_source'
        deadline = frame.get('wall_deadline_monotonic')
        if deadline is not None and deadline-time.perf_counter() < .150:
            return [], 'retrospective_route_scan_skipped_budget'
        self.fallback_attempted.add(track_id)
        snapshot = candidates[-1]
        old_rgb = snapshot['frame']['rgb']
        old_bus = {'track_id': track_id, 'bus_box': snapshot['bus_box']}
        historical = self._accepted_displays(
            self.detector.route_regions(old_rgb, snapshot['bus_box']),
            old_bus, old_rgb, snapshot['buses'])
        mapped = []
        for old_box, _ in historical:
            projected = self._project_box(old_box, snapshot['bus_box'], selected['bus_box'])
            if projected is not None:
                mapped.append(projected)
        return self._accepted_displays(mapped, selected, rgb), 'retrospective_route_scan_used'

    def _maybe_seed_selected(self, frame, selected):
        track_id = selected['track_id']
        ring = self.raw_frames.get(track_id)
        if not ring:
            return None
        now = frame['timestamp_s']
        if not any(0 < now-item['frame']['timestamp_s'] <= .25+1e-6 for item in ring):
            self.raw_frames.pop(track_id, None)
            for display in self.displays.values():
                display.pop('retrospective_buffer', None)
            return 'retrospective_source_expired'
        if not self.temporal:
            self.raw_frames.pop(track_id, None)
            return None
        if not self.displays or all('retrospective_buffer' in value for value in self.displays.values()):
            return None
        deadline = frame.get('wall_deadline_monotonic')
        if deadline is not None and deadline-time.perf_counter() < .080:
            return 'retrospective_ocr_skipped_budget'
        self._seed_selected_displays(now, selected)
        return None

    def _refresh_selected_displays(self, frame, selected, current_scan=True):
        rgb, now = frame['rgb'], frame['timestamp_s']
        track_id = selected['track_id']
        accepted = self._accepted_displays(
            self.detector.route_regions(rgb, selected['bus_box']), selected, rgb) if current_scan else []
        fallback_reason = None
        if not accepted and self.raw_frames.get(track_id):
            accepted, fallback_reason = self._fallback_display_boxes(frame, selected)
        fresh = {}
        used = set()
        for detector_box, box in accepted:
            matches = [key for key, value in self.displays.items()
                       if value['track_id'] == track_id and iou(value['box'], box) >= .3]
            if len(matches) == 1 and matches[0] not in used:
                key = matches[0]
                value = self.displays[key]
                old = value['box']
                ratio = ((box[2]-box[0])*(box[3]-box[1]) /
                         max(1, (old[2]-old[0])*(old[3]-old[1])))
                if not .65 <= ratio <= 1.55:
                    value = {'buffer': DisplayBuffer()}
            else:
                self.next_display += 1
                key = self.next_display
                value = {'buffer': DisplayBuffer()}
            used.add(key)
            fresh[key] = {**value, 'track_id': track_id, 'box': tuple(box),
                          'detector_box': tuple(detector_box), 'bus_box': selected['bus_box']}
        self.displays = fresh
        seed_reason = self._maybe_seed_selected(frame, selected)
        return fallback_reason or seed_reason

    def _propagate_preselection_displays(self, rgb, active):
        keys = list(self.displays)
        boxes = propagate_boxes(self.previous, rgb, [self.displays[key]['box'] for key in keys]) if keys else []
        for key, box in zip(keys, boxes):
            display = self.displays[key]
            bus = active.get(display['track_id'])
            if bus is None or box is None or not self._allowed(box, bus, rgb):
                del self.displays[key]
            else:
                display['box'] = box

    def _process_preselection(self, frame):
        begin = time.perf_counter()
        now, rgb = frame['timestamp_s'], frame['rgb']
        detection_due = now-self.last_detection >= self.detection_interval-1e-6
        selected = None
        fallback_reason = None
        if detection_due:
            buses = self.detector.buses(rgb)
            stable, pan = self._camera_state(now, rgb, buses)
            previous_focus = self.selector.focus
            self.buses = buses
            selected = self.selector.update(now, buses, rgb.shape, stable, camera_pan=pan)
            self.last_detection = now
            self.detection_image = rgb
            epoch_changed = self.epoch != self.selector.epoch
            if epoch_changed:
                # The first selection may inherit only its own continuous track.
                # A lost/overlapping/reassigned focus never inherits old evidence.
                keep = selected['track_id'] if previous_focus is None and selected is not None else None
                self._clear(keep)
            self.context_stable = bool(self.selector.profile is not None and (stable or pan) and
                                       not (epoch_changed and selected is None))
            active = self._preselection_buses(now, selected, True)
            if selected is not None and selected['track_id'] in active:
                fallback_reason = self._refresh_selected_displays(frame, selected)
            else:
                self.displays.clear()
        else:
            if self.selector.focus is not None:
                selected = next((bus for bus in self.buses
                                 if bus['track_id'] == self.selector.focus), None)
            active = self._preselection_buses(now, selected, False)
            if selected is not None:
                self._propagate_preselection_displays(rgb, active)
                if selected['track_id'] in active:
                    if not self.displays and self.raw_frames.get(selected['track_id']):
                        fallback_reason = self._refresh_selected_displays(frame, selected, current_scan=False)
                    else:
                        fallback_reason = self._maybe_seed_selected(frame, selected)
            else:
                self.displays.clear()

        if selected is None:
            for track_id, bus in active.items():
                self._remember_preselection_frame(track_id, frame, bus, detection_due)
        elif selected['track_id'] in active:
            self.sources[(selected['track_id'], frame['source_frame_index'])] = frame
        self.sources = {key: value for key, value in self.sources.items()
                        if key[0] in active and now-value['timestamp_s'] <= .4+1e-6}

        predictions = []
        quality = ([{'display_id': None, 'track_id': selected['track_id'], 'reason': fallback_reason}]
                   if fallback_reason and selected is not None else [])
        earliest_source = self._earliest_live_source(frame)
        if selected is None:
            quality.extend({'display_id': None, 'track_id': track_id,
                            'reason': 'source_ring_buffered'} for track_id in active)
        for key, display in self.displays.items():
            track_id = display['track_id']
            x, y, z, b = display['box']
            crop = rgb[y:b, x:z]
            display['buffer'].add(frame, crop, display['box'])
            if self.temporal:
                best, reason = display['buffer'].best(now, earliest_source)
            else:
                best = {**frame, 'crop': crop, 'box': display['box'], 'relative_stroke_coverage': None}
                reason = 'current_frame_ablation'
            quality.append({'display_id': key, 'track_id': track_id, 'reason': reason})
            retro = display.get('retrospective_buffer')
            if best is not None:
                display.pop('retrospective_buffer', None)
                self.raw_frames.pop(track_id, None)
            if best is None and retro is not None:
                deadline = frame.get('wall_deadline_monotonic')
                if deadline is None or deadline-time.perf_counter() >= .080:
                    # The current real crop supplies temporal span only. The
                    # normal buffer keeps its own copy and its future OCR chance.
                    if frame['source_frame_index'] > retro.last_index:
                        retro.add(frame, crop, display['box'])
                    best, reason = retro.best(now, earliest_source)
                    if best is not None and best['source_frame_index'] == frame['source_frame_index']:
                        best, reason = None, 'current_source_reserved_for_normal_buffer'
                        display.pop('retrospective_buffer', None)
                        self.raw_frames.pop(track_id, None)
                    elif best is not None:
                        display.pop('retrospective_buffer', None)
                        self.raw_frames.pop(track_id, None)
                    elif not any(item['source_frame_index'] < frame['source_frame_index'] and
                                 now-item['timestamp_s'] <= .25+1e-6 for item in retro.frames):
                        display.pop('retrospective_buffer', None)
                        self.raw_frames.pop(track_id, None)
                    quality.append({'display_id': key, 'track_id': track_id,
                                    'reason': 'retrospective_' + reason})
                else:
                    quality.append({'display_id': key, 'track_id': track_id,
                                    'reason': 'retrospective_ocr_skipped_budget'})
            if best is None:
                continue
            source = self.sources.get((track_id, best['source_frame_index']))
            if source is None or best['timestamp_s'] > now or now-best['timestamp_s'] > .25+1e-6:
                continue
            pred = self.recognizer.predict_batch([Image.fromarray(best['crop'])])[0]
            rejection = text_rejection(pred.text, best['box'])
            item = {'epoch': self.epoch, 'track_id': track_id, 'display_id': key,
                    'surface': 'unknown',
                    'source_frame_index': best['source_frame_index'], 'source_pts': best['source_pts'],
                    'source_pts_s': best['source_pts_s'], 'timestamp_s': best['timestamp_s'],
                    'available_at_s': now, 'text_box': best['box'], 'bus_box': display['bus_box'],
                    'detector_box': display['detector_box'],
                    'relative_stroke_coverage': best['relative_stroke_coverage'],
                    **asdict(pred), 'eligible': not rejection, 'rejection_reason': rejection}
            item['context_checks'] = []
            for box in context_boxes(item, source['rgb'].shape):
                a, c, d, e = box
                cp = self.recognizer.predict_batch([Image.fromarray(source['rgb'][c:e, a:d])])[0]
                rejection = context_rejection(item['text'], cp.text, cp.token_scores)
                item['context_checks'].append({**asdict(cp), 'box': box, 'rejection_reason': rejection})
                if rejection:
                    item['eligible'] = False
                    item['rejection_reason'] = rejection
            if (track_id, item['source_frame_index']) in self.observed_sources:
                item['eligible'] = False
                item['rejection_reason'] = 'source_frame_already_counted_on_this_bus'
            predictions.append(item)
        self.observed_sources.update((p['track_id'], p['source_frame_index']) for p in predictions)
        oldest_index = min((key[1] for key in self.sources), default=frame['source_frame_index'])
        self.observed_sources = {key for key in self.observed_sources if key[0] in active and key[1] >= oldest_index}

        decision = None
        if selected is not None and selected['track_id'] in active:
            track_id = selected['track_id']
            decision = self.matcher.update(track_id, now, predictions, selected['bus_box'],
                                           complete=bool(predictions))
        self.previous = rgb
        self.detector.sync()
        elapsed = (time.perf_counter()-begin)*1000
        return {'timestamp_s': now, 'source_frame_index': frame['source_frame_index'],
                'source_pts': frame['source_pts'], 'source_pts_s': frame['source_pts_s'],
                'epoch': self.epoch, 'selected_track_id': self.selector.focus,
                'selection_reason': self.selector.reason, 'bus_detection_ran': detection_due,
                'camera_diagnostics': self.camera_diagnostics if detection_due else None,
                'profile_diagnostics': self.selector.candidate_diagnostics if detection_due else None,
                'buses': self.buses, 'quality': quality, 'predictions': predictions,
                'decision': decision, 'pipeline_ms': elapsed}

    def process(self, frame):
        if self.preselection_collection:
            return self._process_preselection(frame)
        return self._process_selection_first(frame)

    def _process_selection_first(self, frame):
        begin = time.perf_counter()
        now, rgb = frame['timestamp_s'], frame['rgb']
        detection_due = now-self.last_detection >= self.detection_interval-1e-6
        selected = None
        if detection_due:
            buses = self.detector.buses(rgb)
            stable, pan = self._camera_state(now, rgb, buses)
            self.buses = buses
            selected = self.selector.update(now, buses, rgb.shape, stable, camera_pan=pan)
            self.last_detection = now
            self.detection_image = rgb
            if self.epoch != self.selector.epoch:
                self._clear()
            if selected is None:
                self.displays.clear()
            else:
                # Only the selected bus ever enters route detection / OCR.
                boxes = self.detector.route_regions(rgb, selected['bus_box'])
                used = set()
                fresh = {}
                for detector_box in boxes:
                    if not self._allowed(detector_box, selected, rgb):
                        continue
                    box = display_crop_box(detector_box,selected['bus_box'])
                    if not self._allowed(box, selected, rgb):
                        continue
                    matches = [key for key, value in self.displays.items()
                               if iou(value['box'], box) >= .3]
                    if len(matches) == 1 and matches[0] not in used:
                        key = matches[0]
                        value = self.displays[key]
                        # A shrinking partial detector box must not inherit the
                        # previous full-display envelope through arbitrary resize.
                        old = value['box']
                        ratio = (box[2]-box[0])*(box[3]-box[1])/max(1,(old[2]-old[0])*(old[3]-old[1]))
                        if not .65 <= ratio <= 1.55:
                            value = {'buffer': DisplayBuffer()}
                    else:
                        self.next_display += 1
                        key = self.next_display
                        value = {'buffer': DisplayBuffer()}
                    used.add(key)
                    fresh[key] = {**value, 'box': tuple(box), 'detector_box':tuple(detector_box), 'bus_box': selected['bus_box']}
                self.displays = fresh
        elif self.selector.focus is not None:
            selected = next((b for b in self.buses if b['track_id'] == self.selector.focus), None)
            keys = list(self.displays)
            boxes = propagate_boxes(self.previous, rgb, [self.displays[key]['box'] for key in keys]) if keys else []
            for key, box in zip(keys, boxes):
                if box is None or not self._allowed(box, selected, rgb):
                    del self.displays[key]
                else:
                    self.displays[key]['box'] = box
        if selected is None:
            self.sources.clear()
        else:
            self.sources[frame['source_frame_index']] = frame
            self.sources = {key:value for key,value in self.sources.items() if now-value['timestamp_s'] <= .4+1e-6}
        predictions = []
        quality = []
        earliest_source = self._earliest_live_source(frame)
        for key, display in self.displays.items():
            x,y,z,b = display['box']
            crop = rgb[y:b,x:z]
            display['buffer'].add(frame, crop, display['box'])
            if self.temporal:
                best, reason = display['buffer'].best(now, earliest_source)
            else:
                best = {**frame, 'crop':crop, 'box':display['box'], 'relative_stroke_coverage':None}
                reason = 'current_frame_ablation'
            quality.append({'display_id':key, 'reason':reason})
            if best is None:
                continue
            source = self.sources.get(best['source_frame_index'])
            if source is None or best['timestamp_s'] > now or now-best['timestamp_s'] > .25+1e-6:
                continue
            pred = self.recognizer.predict_batch([Image.fromarray(best['crop'])])[0]
            rejection = text_rejection(pred.text, best['box'])
            item = {'epoch':self.epoch, 'track_id':selected['track_id'], 'display_id':key,
                    'surface':'unknown',
                    'source_frame_index':best['source_frame_index'], 'source_pts':best['source_pts'],
                    'source_pts_s':best['source_pts_s'], 'timestamp_s':best['timestamp_s'],
                    'available_at_s':now, 'text_box':best['box'], 'bus_box':display['bus_box'],
                    'detector_box':display['detector_box'],
                    'relative_stroke_coverage':best['relative_stroke_coverage'],
                    **asdict(pred), 'eligible':not rejection, 'rejection_reason':rejection}
            item['context_checks'] = []
            for box in context_boxes(item, source['rgb'].shape):
                a,c,d,e = box
                cp = self.recognizer.predict_batch([Image.fromarray(source['rgb'][c:e,a:d])])[0]
                rejection = context_rejection(item['text'],cp.text,cp.token_scores)
                item['context_checks'].append({**asdict(cp),'box':box,'rejection_reason':rejection})
                if rejection:
                    item['eligible'] = False
                    item['rejection_reason'] = rejection
            if item['source_frame_index'] in self.observed_sources:
                item['eligible'] = False
                item['rejection_reason'] = 'source_frame_already_counted_on_this_bus'
            predictions.append(item)
        self.observed_sources.update(p['source_frame_index'] for p in predictions)
        oldest_index = min(self.sources, default=frame['source_frame_index'])
        self.observed_sources = {i for i in self.observed_sources if i >= oldest_index}
        decision = None
        if selected is not None:
            decision = self.matcher.update(selected['track_id'],now,predictions,selected['bus_box'], complete=bool(predictions))
        self.previous = rgb
        self.detector.sync()
        elapsed = (time.perf_counter()-begin)*1000
        return {'timestamp_s':now,'source_frame_index':frame['source_frame_index'],
                'source_pts':frame['source_pts'],'source_pts_s':frame['source_pts_s'],
                'epoch':self.epoch,'selected_track_id':self.selector.focus,
                'selection_reason':self.selector.reason,'bus_detection_ran':detection_due,
                'camera_diagnostics':self.camera_diagnostics if detection_due else None,
                'profile_diagnostics':self.selector.candidate_diagnostics if detection_due else None,
                'buses':self.buses,'quality':quality,'predictions':predictions,
                'decision':decision,'pipeline_ms':elapsed}
