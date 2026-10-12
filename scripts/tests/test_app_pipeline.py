"""통합앱 버스 번호 인식 경로(bus_route.pipeline)의 계약 테스트.

gildongmu-app tests/test_bus_inference.py, test_bus_observed_routes.py,
test_led_diagnostics.py, test_bus_continuity.py, test_bus_evidence_window.py와
같은 내용을 unittest로 옮겼다. 모델 대신 가짜를 쓴다.
"""
from dataclasses import dataclass
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from bus_route.base import InferenceContext, ModelSpec
from bus_route.continuity import BusTrackContinuity
from bus_route.led_diagnostics import led_row_diagnostics
from bus_route.observed import ObservedRouteMatcher
from bus_route.pipeline import BusPipeline
from bus_route.target import TargetMatcher


@dataclass(frozen=True)
class Prediction:
    """Model-free substitute for the PARSeq output contract."""
    text: str
    token_scores: list[float]
    elapsed_ms: float


class FakeDetector:
    def __init__(self):
        self.tracks = [(1, (10, 10, 100, 150)), (2, (110, 10, 195, 150))]
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def sync(self):
        pass

    def buses(self, rgb):
        return [{'track_id': track_id, 'bus_box': box, 'bus_score': .95}
                for track_id, box in self.tracks]

    def text_regions(self, rgb, bus_box):
        x, y, _, _ = bus_box
        return bus_box, [(x+20, y+20, x+70, y+35)]


class FakeRecognizer:
    def __init__(self, *, fail=False, score=.995):
        self.fail = fail
        self.score = score

    def predict_batch(self, images):
        if self.fail:
            raise RuntimeError('OCR unavailable')
        values = []
        for image in images:
            text = '701' if float(np.asarray(image).mean()) > 120 else '604'
            values.append(Prediction(text, [self.score]*(len(text)+1), 1.0))
        return values


def make_frame():
    frame = np.full((180, 220, 3), 50, np.uint8)
    frame[30:45, 30:80] = 80
    frame[30:45, 130:180] = 190
    return frame


def make_pipeline():
    spec = ModelSpec(id='bus-route-display-b', mode='bus', name='B', version='B',
                     weights=Path('/unused/route-display-b.pt'))
    pipeline = BusPipeline(spec)
    pipeline.detector = FakeDetector()
    pipeline.recognizer = FakeRecognizer()
    return pipeline


class BusInferenceTests(unittest.TestCase):
    def test_per_bus_number_and_target_decision_are_separate(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        frame = make_frame()
        for frame_id, ms in enumerate((1000, 1200, 1400), 1):
            result = pipeline.infer(frame, InferenceContext('one', frame_id, ms))
        event = result['event']
        self.assertEqual(event['type'], 'bus_detection')
        self.assertEqual(event['target_route'], '604')
        self.assertEqual([(m['track_id'], m['state']) for m in event['matches']],
                         [(1, 'matched_candidate')])
        self.assertEqual(event['bus_number'], '604')
        self.assertTrue(event['is_target'])
        self.assertEqual([b['observations'][0]['text'] for b in event['buses']],
                         ['604', '701'])
        self.assertEqual(event['buses'][1]['decision']['state'], 'unverified')
        self.assertEqual(sum(d['class_name'] == 'bus' for d in result['detections']), 2)
        for detection in result['detections']:
            box = detection['box']
            self.assertTrue(0 <= box['x1'] < box['x2'] <= 1)
            self.assertTrue(0 <= box['y1'] < box['y2'] <= 1)

    def test_ocr_failure_keeps_bus_boxes_and_does_not_announce(self):
        pipeline = make_pipeline()
        pipeline.recognizer = FakeRecognizer(fail=True)
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        with self.assertLogs('bus_route.pipeline', level='ERROR'):
            result = pipeline.infer(make_frame(), InferenceContext('one', 1, 1000))
        self.assertEqual([d['class_name'] for d in result['detections']], ['bus', 'bus'])
        self.assertEqual(result['event']['matches'], [])
        self.assertIsNone(result['event']['bus_number'])
        self.assertFalse(result['event']['is_target'])
        self.assertTrue(result['event']['errors'][0].startswith('ocr_failed:'))

    def test_new_session_clears_track_history_and_target(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        for frame_id, ms in enumerate((1000, 1200, 1400), 1):
            pipeline.infer(make_frame(), InferenceContext('one', frame_id, ms))
        pipeline.close_session('one')
        pipeline.configure_target_route('701')
        pipeline.reset_session('two')
        result = pipeline.infer(make_frame(), InferenceContext('two', 1, 1600))
        self.assertEqual([(m['track_id'], m['route_number'], m['state'])
                          for m in result['event']['matches']],
                         [(2, '701', 'recognized_single')])
        self.assertEqual(result['event']['buses'][0]['decision']['state'], 'unverified')
        self.assertEqual(pipeline.detector.reset_count, 3)

    def test_no_target_still_recognizes_numbers_and_nonmonotonic_frame_is_held(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route(None)
        pipeline.reset_session('one')
        result = pipeline.infer(make_frame(), InferenceContext('one', 1, 1000))
        self.assertEqual(result['event']['matches'], [])
        self.assertEqual(result['event']['buses'][0]['observations'][0]['text'], '604')
        self.assertEqual(result['event']['buses'][0]['decision']['state'], 'not_configured')
        pipeline.configure_target_route('604')
        pipeline.reset_session('two')
        pipeline.infer(make_frame(), InferenceContext('two', 1, 1200))
        late = pipeline.infer(make_frame(), InferenceContext('two', 2, 1100))
        self.assertEqual(late['event']['matches'], [])
        self.assertEqual(late['event']['buses'][0]['decision']['reason'],
                         'nonmonotonic_capture_time')

    def test_omitted_valid_region_prevents_target_announcement(self):
        class TooManyDetector(FakeDetector):
            def __init__(self):
                super().__init__()
                self.tracks = [(1, (10, 10, 100, 150))]

            def text_regions(self, rgb, bus_box):
                return bus_box, [(30, 30, 80, 45)]*25

        pipeline = make_pipeline()
        pipeline.detector = TooManyDetector()
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        result = pipeline.infer(make_frame(), InferenceContext('one', 1, 1000))
        bus = result['event']['buses'][0]
        self.assertTrue(bus['candidate_truncated'])
        self.assertEqual(len(bus['observations']), 24)
        self.assertEqual(bus['decision']['state'], 'hold')
        self.assertEqual(result['event']['matches'], [])


BOX = (0, 0, 100, 100)


def reading(text, score=.99, eligible=True):
    return {'text': text, 'token_scores': [score] * (len(text) + 1), 'eligible': eligible}


def observe(matcher, timestamp, observations, track=1, box=BOX, complete=True, target='7011'):
    return matcher.update(track, timestamp, observations, box, target=target, complete=complete)


class ObservedRouteTests(unittest.TestCase):
    """Speech evidence for another visible bus stays separate from the target match."""

    def test_other_bus_requires_repeated_current_readings_and_keeps_target_contract(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('604')
        pipeline.reset_session('session')
        for frame_id, ms in enumerate((1000, 1200, 1400), 1):
            result = pipeline.infer(make_frame(), InferenceContext('session', frame_id, ms, .25))
            other = [item for item in result['event']['recognized_routes'] if not item['is_target']]
            self.assertEqual(len(other), 1 if frame_id == 3 else 0)
        self.assertEqual(other[0]['route_number'], '701')
        self.assertEqual(other[0]['state'], 'matched_candidate')
        self.assertEqual(result['event']['bus_number'], '604')
        self.assertEqual([item['route_number'] for item in result['event']['matches']], ['604'])
        pipeline.reset_session('next')
        result = pipeline.infer(make_frame(), InferenceContext('next', 1, 1600, .25))
        self.assertTrue(all(item['is_target'] for item in result['event']['recognized_routes']))

    def test_partial_target_weak_ambiguous_or_invalid_readings_do_not_speak(self):
        cases = [
            [reading('701')], [reading('11')], [reading('7011')],
            [reading('1')], [reading('711')], [reading('71')],
            [reading('173', .89)], [reading('173', eligible=False)],
            [reading('173'), reading('174')], [reading('173'), reading('7011')],
            [reading('전화')], [reading('173', float('nan'))],
        ]
        for observations in cases:
            with self.subTest(observations=[o['text'] for o in observations]):
                matcher = ObservedRouteMatcher()
                for timestamp in (1, 1.2, 1.4, 1.6):
                    self.assertEqual(observe(matcher, timestamp, observations), [])

    def test_lost_current_evidence_cannot_confirm_another_bus(self):
        for invalid in ('duplicate_time', 'gap', 'jump', 'incomplete', 'untracked', 'missing'):
            with self.subTest(invalid=invalid):
                matcher = ObservedRouteMatcher()
                for timestamp in (1, 1.2):
                    self.assertEqual(observe(matcher, timestamp, [reading('173')]), [])
                kwargs = {}
                timestamp = 1.4
                observations = [reading('173')]
                if invalid == 'duplicate_time': timestamp = 1.2
                if invalid == 'gap': timestamp = 2
                if invalid == 'jump': kwargs['box'] = (200, 200, 300, 300)
                if invalid == 'incomplete': kwargs['complete'] = False
                if invalid == 'untracked': kwargs['track'] = None
                if invalid == 'missing': observations = []
                self.assertEqual(observe(matcher, timestamp, observations, **kwargs), [])

    def test_short_reading_after_longer_route_is_not_a_new_confirmed_number(self):
        matcher = ObservedRouteMatcher()
        observe(matcher, 1, [reading('1711')])
        for timestamp in (1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3):
            self.assertEqual(observe(matcher, timestamp, [reading('171')]), [])

    def test_longer_number_context_expires_when_track_is_lost(self):
        matcher = ObservedRouteMatcher()
        observe(matcher, 1, [reading('1711')])
        for timestamp in (2, 2.2):
            self.assertEqual(observe(matcher, timestamp, [reading('171')]), [])
        self.assertEqual(observe(matcher, 2.4, [reading('171')])[0]['route_number'], '171')

    def test_nonmonotonic_frames_cannot_publish_observed_confirmation(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('7011')
        pipeline.reset_session('session')
        for frame_id, ms in enumerate((1000, 1200, 1400), 1):
            pipeline.infer(make_frame(), InferenceContext('session', frame_id, ms, .25))
        result = pipeline.infer(make_frame(), InferenceContext('session', 4, 1300, .25))
        self.assertEqual(result['event']['recognized_routes'], [])


AMBER = (255, 170, 30)


def led_crop(lit_rows):
    crop = np.full((24, 60, 3), 20, np.uint8)
    for row in lit_rows:
        crop[row, 5:55] = AMBER
    return crop


class LedDiagnosticsTests(unittest.TestCase):
    """Diagnostic-only LED scan-gap measurement on synthetic route-number crops."""

    def test_fully_lit_rows_are_not_suspected(self):
        result = led_row_diagnostics(led_crop(range(3, 21)))
        self.assertEqual(result['row_occupancy'], 1.0)
        self.assertFalse(result['scan_gap_suspected'])

    def test_dot_matrix_gaps_stay_above_threshold(self):
        # A clean dot matrix can leave a dark line between most LED rows.
        rows = [r for r in range(3, 21) if r % 5 != 0]
        self.assertFalse(led_row_diagnostics(led_crop(rows))['scan_gap_suspected'])

    def test_few_scan_lines_are_suspected(self):
        result = led_row_diagnostics(led_crop([3, 4, 12, 20]))
        self.assertLess(result['row_occupancy'], .8)
        self.assertTrue(result['scan_gap_suspected'])

    def test_unsaturated_or_tiny_crops_make_no_claim(self):
        white = np.full((24, 60, 3), 20, np.uint8)
        white[3:21, 5:55] = 240
        self.assertIsNone(led_row_diagnostics(white))
        self.assertIsNone(led_row_diagnostics(led_crop([1])[:4]))
        self.assertIsNone(led_row_diagnostics(np.zeros((24, 60, 3), np.uint8)))

    def test_small_span_is_measured_but_not_judged(self):
        result = led_row_diagnostics(led_crop([3, 6, 9, 12]))
        self.assertLess(result['row_occupancy'], .8)
        self.assertIsNone(result['scan_gap_suspected'])


class PipelineLedDiagnosticsTests(unittest.TestCase):
    def run_once(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        return pipeline.infer(make_frame(), InferenceContext('one', 1, 1000))['event']

    def test_observations_carry_diagnostics_without_changing_ocr(self):
        event = self.run_once()
        observations = [o for b in event['buses'] for o in b['observations']]
        self.assertEqual([o['text'] for o in observations], ['604', '701'])
        self.assertTrue(all('led_diagnostics' in o for o in observations))

    def test_diagnostic_failure_keeps_ocr_result(self):
        with mock.patch('bus_route.pipeline.led_row_diagnostics', side_effect=RuntimeError('boom')):
            with self.assertLogs('bus_route.pipeline', level='ERROR'):
                event = self.run_once()
        observations = [o for b in event['buses'] for o in b['observations']]
        self.assertEqual([o['text'] for o in observations], ['604', '701'])
        self.assertTrue(all(o['led_diagnostics'] is None for o in observations))
        self.assertEqual(event['errors'], [])



BUS_BOX = (10, 10, 100, 150)


def track(tracker, timestamp, entries):
    buses = [dict(track_id=track_id, bus_box=box) for track_id, box in entries]
    tracker.update(buses, timestamp)
    return buses


class BusContinuityTests(unittest.TestCase):
    """Missing bus IDs can recover only through separate, unambiguous geometry."""

    def test_one_missing_id_is_not_enough_and_same_time_is_not_a_second_observation(self):
        tracker = BusTrackContinuity()
        self.assertIsNone(track(tracker, 1, [(None, BUS_BOX)])[0]['track_id'])
        self.assertIsNone(track(tracker, 1, [(None, BUS_BOX)])[0]['track_id'])
        self.assertIsNone(track(tracker, .9, [(None, BUS_BOX)])[0]['track_id'])
        recovered, = track(tracker, 1.4, [(None, BUS_BOX)])
        self.assertLess(recovered['track_id'], 0)
        self.assertIsNone(recovered['detector_track_id'])
        self.assertEqual(recovered['track_source'], 'provisional')

    def test_detector_id_changes_retain_unique_recent_identity(self):
        tracker = BusTrackContinuity()
        self.assertEqual(track(tracker, 1, [(7, BUS_BOX)])[0]['track_id'], 7)
        missing, = track(tracker, 1.4, [(None, BUS_BOX)])
        self.assertEqual(missing['track_id'], 7)
        changed, = track(tracker, 1.8, [(12, BUS_BOX)])
        self.assertEqual(changed['track_id'], 7)
        self.assertEqual(changed['detector_track_id'], 12)
        self.assertEqual(changed['track_source'], 'continuity')

    def test_new_detector_id_can_take_over_a_provisional_vehicle(self):
        tracker = BusTrackContinuity()
        track(tracker, 1, [(None, BUS_BOX)])
        provisional = track(tracker, 1.4, [(None, BUS_BOX)])[0]['track_id']
        self.assertEqual(track(tracker, 1.8, [(9, BUS_BOX)])[0]['track_id'], provisional)

    def test_expired_or_different_vehicle_cannot_inherit_identity(self):
        for later, box in [(2.01, BUS_BOX), (1.4, (110, 10, 200, 150))]:
            with self.subTest(later=later, box=box):
                tracker = BusTrackContinuity()
                track(tracker, 1, [(7, BUS_BOX)])
                self.assertEqual(track(tracker, later, [(9, box)])[0]['track_id'], 9)

    def test_overlapping_untracked_buses_are_never_promoted(self):
        tracker = BusTrackContinuity()
        for t in (1, 1.4, 1.8, 2.2):
            buses = track(tracker, t, [(None, BUS_BOX), (None, (12, 12, 102, 152))])
            self.assertTrue(all(bus['track_id'] is None for bus in buses))

    def test_two_possible_previous_vehicles_do_not_share_their_evidence(self):
        tracker = BusTrackContinuity()
        track(tracker, 1, [(7, BUS_BOX), (8, (12, 12, 102, 152))])
        self.assertEqual(track(tracker, 1.4, [(9, BUS_BOX)])[0]['track_id'], 9)

    def test_two_new_vehicles_cannot_both_claim_one_previous_vehicle(self):
        tracker = BusTrackContinuity()
        track(tracker, 1, [(7, BUS_BOX)])
        buses = track(tracker, 1.4, [(9, BUS_BOX), (10, BUS_BOX)])
        self.assertEqual({b['track_id'] for b in buses}, {9, 10})

    def test_two_live_detector_ids_previously_linked_split_without_duplicate_identity(self):
        tracker = BusTrackContinuity()
        track(tracker, 1, [(7, BUS_BOX)])
        track(tracker, 1.4, [(9, BUS_BOX)])
        buses = track(tracker, 1.8, [(7, BUS_BOX), (9, (110, 10, 200, 150))])
        self.assertEqual({b['track_id'] for b in buses}, {7, 9})

    def test_reset_and_expiration_drop_all_old_aliases(self):
        tracker = BusTrackContinuity()
        track(tracker, 1, [(7, BUS_BOX)])
        track(tracker, 1.4, [(9, BUS_BOX)])
        track(tracker, 3, [])
        self.assertEqual(tracker._tracks, {})
        self.assertEqual(tracker._aliases, {})
        tracker.reset()
        self.assertEqual(track(tracker, 1, [(9, BUS_BOX)])[0]['track_id'], 9)

    def test_pipeline_reads_an_untracked_bus_after_two_frames_and_preserves_ocr_metadata(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        pipeline.detector.tracks = [(None, BUS_BOX)]
        first = pipeline.infer(make_frame(), InferenceContext('one', 1, 1000))['event']
        self.assertEqual(first['matches'], [])
        second = pipeline.infer(make_frame(), InferenceContext('one', 2, 1400))['event']
        match, = second['matches']
        self.assertEqual(match['state'], 'recognized_single')
        self.assertLess(match['track_id'], 0)
        bus, = second['buses']
        self.assertIsNone(bus['detector_track_id'])
        self.assertEqual(bus['track_source'], 'provisional')
        self.assertEqual(bus['observations'][0]['track_id'], match['track_id'])
        self.assertIsNone(bus['observations'][0]['detector_track_id'])
        pipeline.reset_session('two')
        self.assertEqual(pipeline.infer(make_frame(), InferenceContext('two', 1, 1800))['event']['matches'], [])

    def test_changed_detector_id_accumulates_independent_reads_on_the_same_vehicle(self):
        pipeline = make_pipeline()
        pipeline.recognizer.score = .95  # Must reach the original three-read policy.
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        for frame, (tid, timestamp) in enumerate([(7, 1000), (9, 1200), (9, 1400)], 1):
            pipeline.detector.tracks = [(tid, BUS_BOX)]
            event = pipeline.infer(make_frame(), InferenceContext('one', frame, timestamp))['event']
            self.assertEqual(bool(event['matches']), frame == 3)
        self.assertEqual(event['matches'][0]['state'], 'matched_candidate')
        self.assertEqual(event['matches'][0]['track_id'], 7)

    def test_pipeline_duplicate_or_older_capture_cannot_create_provisional_evidence(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        pipeline.detector.tracks = [(None, BUS_BOX)]
        for frame, capture in enumerate([1000, 1000, 900], 1):
            self.assertEqual(pipeline.infer(make_frame(), InferenceContext('one', frame, capture))['event']['matches'], [])
        self.assertTrue(pipeline.infer(make_frame(), InferenceContext('one', 4, 1400))['event']['matches'])

    def test_promoting_an_id_cannot_override_ineligible_ocr(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        pipeline.detector.tracks = [(None, BUS_BOX)]
        # A region on the bus search boundary stays ineligible even after ID recovery.
        pipeline.detector.text_regions = lambda rgb, box: (box, [(10, 30, 80, 45)])
        for frame, capture in enumerate([1000, 1400, 1800], 1):
            event = pipeline.infer(make_frame(), InferenceContext('one', frame, capture))['event']
            self.assertEqual(event['matches'], [])
        self.assertLess(event['buses'][0]['track_id'], 0)
        self.assertIs(event['buses'][0]['observations'][0]['eligible'], False)

    def test_promoting_an_id_cannot_confirm_an_incomplete_scan(self):
        pipeline = make_pipeline()
        pipeline.configure_target_route('604')
        pipeline.reset_session('one')
        pipeline.detector.tracks = [(None, BUS_BOX)]
        pipeline.detector.text_regions = lambda rgb, box: (box, [(30, 30, 80, 45)] * 25)
        for frame, capture in enumerate([1000, 1400, 1800], 1):
            event = pipeline.infer(make_frame(), InferenceContext('one', frame, capture))['event']
            self.assertEqual(event['matches'], [])
        self.assertLess(event['buses'][0]['track_id'], 0)
        self.assertIs(event['buses'][0]['candidate_truncated'], True)
        self.assertEqual(event['buses'][0]['decision']['state'], 'hold')


def window_reading(text='7713', score=.99):
    return dict(text=text, token_scores=[score] * (len(text)+1), eligible=True)


class EvidenceWindowTests(unittest.TestCase):
    """Sparse correct reads accumulate while the same vehicle remains observed."""

    def test_three_reads_across_continuously_observed_bus_can_span_nearly_two_seconds(self):
        for other in (False, True):
            with self.subTest(other=other):
                target = TargetMatcher('7713')
                observed = ObservedRouteMatcher()
                for at in [0, .4, .848, 1.2, 1.6, 1.964]:
                    obs = [window_reading()] if at in [0, .848, 1.964] else []
                    result = (observed.update(7, at, obs, BUS_BOX, target='272', complete=True) if other
                              else target.update(7, at, obs, BUS_BOX))
                if other:
                    self.assertEqual(result[0]['route_number'], '7713')
                    self.assertEqual(result[0]['state'], 'matched_candidate')
                else:
                    self.assertEqual(result['state'], 'matched_candidate')
                    self.assertEqual(result['support_samples'], 3)

    def test_wider_window_does_not_confirm_expired_lost_or_conflicting_evidence(self):
        for problem in ('expired', 'lost', 'incomplete', 'rival', 'missing_current'):
            with self.subTest(problem=problem):
                matcher = TargetMatcher('7713')
                times = [0, .4, .848, 1.2, 1.6, 1.964]
                if problem == 'expired': times[-1] = 2.01
                if problem == 'lost': times = [0, .4, .848, 1.964]
                for at in times:
                    obs = [window_reading()] if at in [0, .848, times[-1]] else []
                    if problem == 'rival' and at == 1.6: obs = [window_reading('7712')]
                    if problem == 'missing_current' and at == times[-1]: obs = []
                    complete = not (problem == 'incomplete' and at == 1.2)
                    result = matcher.update(7, at, obs, BUS_BOX, complete=complete)
                self.assertNotEqual(result['state'], 'matched_candidate')


if __name__ == '__main__':
    unittest.main()
