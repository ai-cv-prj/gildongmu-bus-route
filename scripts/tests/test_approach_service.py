import unittest
from unittest.mock import patch
import numpy as np
import cv2
import tempfile
from pathlib import Path
import subprocess
import imageio_ffmpeg
import time

from bus_route.approach import ApproachProfile, ApproachSelector
from bus_route.service import ApproachPipeline, camera_stable
from bus_route.temporal_display import DisplayBuffer, propagate_boxes, crop_quality, _display_tracking_points
from bus_route.recognizer import Prediction
from bus_route.video import native_frames, video_writer


PROFILE = ApproachProfile(entry_polygon=((0,0),(1,0),(1,1),(0,1)),direction=(0,1))


def bus(t, key=1, offset=0):
    return {'track_id':key,'bus_box':(10+offset,10,50+offset+int(t*30),60+int(t*60))}


def frame(i, t, image=None):
    return {'source_frame_index':i,'timestamp_s':t,'source_pts':i,
            'source_pts_s':t+.09,'rgb':image if image is not None else np.zeros((200,300,3),np.uint8)}


class SelectionTests(unittest.TestCase):
    def select(self, selector):
        for t in [0,.2,.4]:
            result = selector.update(t,[bus(t)],(200,300))
        return result

    def test_unknown_camera_never_selects(self):
        self.assertIsNone(self.select(ApproachSelector()))

    def test_one_target_without_reading_numbers(self):
        selector = ApproachSelector(PROFILE)
        self.assertEqual(self.select(selector)['track_id'],1)

    def test_two_approaches_abstain(self):
        selector = ApproachSelector(PROFILE)
        for t in [0,.2,.4]:
            self.assertIsNone(selector.update(t,[bus(t),bus(t,2,150)],(200,300)))
        self.assertEqual(selector.reason,'ambiguous_approach')

    def test_opposite_lane_outside_profile(self):
        profile = ApproachProfile(entry_polygon=((0,0),(.4,0),(.4,1),(0,1)),direction=(0,1))
        selector = ApproachSelector(profile)
        for t in [0,.2,.4]:
            result = selector.update(t,[bus(t),bus(t,2,150)],(200,300))
        self.assertEqual(result['track_id'],1)

    def test_static_or_receding_bus_not_selected(self):
        for direction in ('static','receding'):
            selector = ApproachSelector(PROFILE)
            for t in [0,.2,.4]:
                self.assertIsNone(selector.update(t,[bus(0 if direction=='static' else .4-t)],(200,300)))

    def test_loss_overlap_jump_and_camera_reset_epoch(self):
        for replacement,stable in [([],True),([bus(.6),bus(.6,2,4)],True),
                                   ([bus(.6,1,150)],True),([bus(.6)],False)]:
            selector = ApproachSelector(PROFILE)
            self.select(selector)
            epoch = selector.epoch
            self.assertIsNone(selector.update(.6,replacement,(200,300),stable))
            self.assertGreater(selector.epoch,epoch)

    def test_id_switch_cannot_inherit_history(self):
        selector = ApproachSelector(PROFILE)
        self.select(selector)
        self.assertIsNone(selector.update(.6,[bus(.6,2)],(200,300)))
        self.assertIsNone(selector.update(.8,[bus(.8,2)],(200,300)))

    def test_duplicate_ids_and_reversed_time_rejected(self):
        selector = ApproachSelector(PROFILE)
        self.assertIsNone(selector.update(0,[bus(0),bus(0)],(200,300)))
        with self.assertRaises(ValueError):
            selector.update(0,[bus(0)],(200,300))

    def test_verified_pan_only_retains_existing_continuous_focus(self):
        selector = ApproachSelector(PROFILE)
        self.assertIsNone(selector.update(0,[bus(0)],(200,300),False,True))
        self.assertEqual(selector.reason,'camera_pan_without_selected_focus')
        for t in (.2,.4,.6):
            selected = selector.update(t,[bus(t)],(200,300))
        self.assertEqual(selected['track_id'],1)
        epoch = selector.epoch
        for t in (.8,1.0):
            selected = selector.update(t,[bus(t)],(200,300),False,True)
            self.assertEqual(selected['track_id'],1)
            self.assertEqual(selector.reason,'selected_approach_pan_hold')
        self.assertEqual(selector.epoch,epoch)

    def test_pan_still_rejects_overlap_and_track_replacement(self):
        for replacement,reason in [([bus(.6),bus(.6,2,4)],'overlapping_identity'),
                                   ([bus(.6,1,150)],'identity_discontinuity'),
                                   ([bus(.6,2)],'track_lost')]:
            selector = ApproachSelector(PROFILE)
            self.select(selector)
            epoch = selector.epoch
            self.assertIsNone(selector.update(.6,replacement,(200,300),False,True))
            self.assertEqual(selector.reason,reason)
            self.assertGreater(selector.epoch,epoch)

    def test_candidate_diagnostics_separate_profile_direction(self):
        selector = ApproachSelector(PROFILE)
        for t in (0,.2,.4):
            # Box grows but its footpoint travels against the configured lane.
            box=(20-round(t*10),40-round(t*20),70+round(t*10),90-round(t*20))
            selector.update(t,[{'track_id':1,'bus_box':box}],(200,300))
        self.assertEqual(selector.candidate_diagnostics[0]['reason'],'insufficient_progress')


class CameraMotionTests(unittest.TestCase):
    def test_measured_motion_and_unmeasurable_background_are_distinct(self):
        blank=np.zeros((240,320,3),np.uint8)
        detail={}
        self.assertFalse(camera_stable(blank,blank,[],detail))
        self.assertEqual(detail['status'],'insufficient_points')
        image=np.random.default_rng(4).integers(0,256,(240,320,3),dtype=np.uint8)
        shifted=cv2.warpAffine(image,np.float32([[1,0,8],[0,1,0]]),(320,240))
        detail={}
        self.assertFalse(camera_stable(image,shifted,[],detail))
        self.assertEqual(detail['status'],'measured_motion')
        self.assertGreaterEqual(detail['flow_count'],10)
        self.assertGreater(detail['normalized_motion'],.018)


class TemporalTests(unittest.TestCase):
    def crop(self, full=True):
        image = np.zeros((48,160,3),np.uint8)
        cv2.putText(image,'606' if full else '60',(8,38),cv2.FONT_HERSHEY_SIMPLEX,1.3,(240,180,30),2)
        return image

    def test_display_feature_roi_preserves_full_frame_tracking_points(self):
        gray = np.random.default_rng(41).integers(0,256,(480,640),dtype=np.uint8)
        for box in ((260,180,340,230),(3,5,100,60),(520,390,630,470)):
            x,y,z,b = box
            margin = max(12,round((b-y)*.6))
            mask = np.zeros(gray.shape,np.uint8)
            mask[max(0,y-margin):min(gray.shape[0],b+margin),
                 max(0,x-margin):min(gray.shape[1],z+margin)] = 255
            mask[max(0,y):min(gray.shape[0],b),
                 max(0,x):min(gray.shape[1],z)] = 0
            original = cv2.goodFeaturesToTrack(gray,50,.01,3,mask=mask)
            optimized = _display_tracking_points(gray,box)
            self.assertIsNotNone(original)
            np.testing.assert_array_equal(optimized,original)

    def test_selects_actual_full_crop_not_high_confidence_or_synthesis(self):
        buffer = DisplayBuffer()
        for i in range(7):
            buffer.add(frame(i,i/30),self.crop(i in [2,5]),(0,0,160,48))
        best, _ = buffer.best(.2)
        self.assertIsNotNone(best)
        self.assertIn(best['source_frame_index'],[2,5])
        np.testing.assert_array_equal(best['crop'],self.crop(True))

    def test_duplicate_and_future_observations_cannot_support(self):
        buffer = DisplayBuffer()
        for i in range(10):
            buffer.add(frame(i,i/30),self.crop(),(0,0,160,48))
        self.assertIsNone(buffer.best(.05)[0])
        with self.assertRaises(ValueError):
            buffer.add(frame(9,.3),self.crop(),(0,0,160,48))

    def test_same_source_not_counted_twice_and_old_buffer_expires(self):
        buffer = DisplayBuffer()
        for i in range(7):
            buffer.add(frame(i,i/30),self.crop(),(0,0,160,48))
        self.assertIsNotNone(buffer.best(.2)[0])
        self.assertIsNone(buffer.best(.2)[0])
        self.assertIsNone(buffer.best(1)[0])

    def test_realtime_deadline_filters_old_sources_without_consuming_fresh_one(self):
        buffer = DisplayBuffer()
        for i,t in enumerate((0,.1,.2)):
            buffer.add(frame(i,t),self.crop(),(0,0,160,48))
        self.assertEqual(buffer.best(.2,earliest_source_s=.21)[1],'no_fresh_evidence')
        best,_=buffer.best(.2,earliest_source_s=.2)
        self.assertEqual(best['source_frame_index'],2)

    def test_low_contrast_abstains(self):
        buffer = DisplayBuffer()
        for i in range(7):
            buffer.add(frame(i,i/30),np.full((48,160,3),80,np.uint8),(0,0,160,48))
        self.assertIsNone(buffer.best(.2)[0])

    def test_stationary_flow_does_not_expand_crop(self):
        image = np.random.default_rng(0).integers(0,256,(160,240,3),dtype=np.uint8)
        box = (30,40,100,80)
        for _ in range(20):
            box = propagate_boxes(image,image,[box])[0]
        self.assertEqual(box,(30,40,100,80))

    def test_shadow_border_does_not_invert_print_foreground(self):
        image = np.full((48,160,3),170,np.uint8)
        image[[0,-1]] = 30
        image[:,[0,-1]] = 30
        cv2.putText(image,'7024',(5,34),cv2.FONT_HERSHEY_SIMPLEX,1,(40,40,40),2)
        self.assertLess(crop_quality(image)['ink_fraction'],.65)


class FakeDetector:
    def __init__(self):
        self.time = 0
        self.route_calls = []
        self.current = None
    def reset(self): pass
    def sync(self): pass
    def buses(self, rgb):
        return self.current if self.current is not None else [bus(self.time),bus(0,2,150)]
    def route_regions(self, rgb, bounds):
        self.route_calls.append(bounds)
        return [(20,20,46,34)]


class FakeRecognizer:
    def __init__(self): self.calls = 0
    def predict_batch(self, images):
        self.calls += len(images)
        return [Prediction('606',[.99]*4,0) for _ in images]


class RelativeDetector(FakeDetector):
    def route_regions(self, rgb, bounds):
        self.route_calls.append(bounds)
        x, y, _, _ = bounds
        return [(x+10, y+10, x+36, y+24)]


class OneMissDetector(RelativeDetector):
    def route_regions(self, rgb, bounds):
        self.route_calls.append(bounds)
        if bounds[2]-bounds[0] >= 52:
            return []
        x, y, _, _ = bounds
        return [(x+10, y+10, x+36, y+24)]


class FilteredCurrentDetector(OneMissDetector):
    def route_regions(self, rgb, bounds):
        if bounds[2]-bounds[0] >= 52:
            self.route_calls.append(bounds)
            return [(bounds[0], bounds[1]+10, bounds[0]+36, bounds[1]+24)]
        return super().route_regions(rgb, bounds)


class RelativeTextDetector(FakeDetector):
    base_bus = (10,10,160,120)
    base_display = (45,30,125,65)

    def buses(self, rgb):
        t = self.time
        return [{'track_id':1, 'bus_box':(10,10,160+round(t*30),120+round(t*30))}]

    def route_regions(self, rgb, bounds):
        self.route_calls.append(bounds)
        return [ApproachPipeline._project_box(self.base_display,self.base_bus,bounds)]


class ColorRecognizer(FakeRecognizer):
    def predict_batch(self, images):
        self.calls += len(images)
        return [Prediction('272' if np.asarray(image)[:, :, 0].mean() > 100 else '606',
                           [.99]*4, 0) for image in images]


class PipelineTests(unittest.TestCase):
    def test_realtime_freshness_uses_source_deadline(self):
        pipe=ApproachPipeline(FakeDetector(),FakeRecognizer(),'606',PROFILE)
        source=frame(1,3.4)
        source['wall_deadline_monotonic']=10.3
        with patch('bus_route.service.time.perf_counter',return_value=10.25):
            self.assertAlmostEqual(pipe._earliest_live_source(source),3.375)
        pipe.realtime_freshness=False
        self.assertIsNone(pipe._earliest_live_source(source))

    def test_only_short_coherent_pan_reaches_selected_bus_ocr(self):
        d,r = FakeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,temporal=False)
        def measured(previous,current,buses,diagnostics=None):
            t=d.time
            delta=[-3.,1.] if t <= .4 else ([-6.,1.] if t <= .6 else
                                                 [-9.,1.] if t <= .8 else [-13.,1.])
            diagnostics.update(status='stable' if t <= .4 else 'measured_motion',
                               median_delta_px=delta,median_residual_px=.5,
                               inlier_fraction_3px=.8)
            return t <= .4
        with patch('bus_route.service.camera_stable',side_effect=measured):
            for i,t in enumerate((0,.2,.4,.6,.8,1.0)):
                d.time=t
                event=pipe.process(frame(i,t))
                if t in (.6,.8):
                    self.assertEqual(event['selected_track_id'],1)
                    self.assertTrue(event['camera_diagnostics']['pan_hold'])
                    self.assertEqual({p['track_id'] for p in event['predictions']},{1})
                if t == 1.0:
                    self.assertIsNone(event['selected_track_id'])
                    self.assertFalse(event['camera_diagnostics']['pan_hold'])

    def test_shake_and_missing_flow_cannot_hold_focus(self):
        for status, residual in [('measured_motion',5.),('insufficient_points',0.)]:
            d,r=FakeDetector(),FakeRecognizer()
            pipe=ApproachPipeline(d,r,'606',PROFILE,temporal=False)
            def measured(previous,current,buses,diagnostics=None):
                diagnostics.update(status=status,median_delta_px=[-6.,1.],
                                   median_residual_px=residual,inlier_fraction_3px=.2)
                return False
            with patch('bus_route.service.camera_stable',side_effect=measured):
                # Establish a focus without bypassing the selected-bus gates.
                with patch.object(pipe,'_camera_state',return_value=(True,False)):
                    for i,t in enumerate((0,.2,.4)):
                        d.time=t
                        pipe.process(frame(i,t))
                pipe.last_camera_delta=[-3.,1.]
                d.time=.6
                event=pipe.process(frame(3,.6))
            self.assertIsNone(event['selected_track_id'])
            self.assertFalse(event['camera_diagnostics']['pan_hold'])

    def test_default_selection_first_preserves_old_ocr_gate(self):
        d,r = FakeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,temporal=False,camera_check=False)
        self.assertFalse(pipe.preselection_collection)
        for i,t in enumerate([0,.2,.4]):
            d.time = t
            event = pipe.process(frame(i,t))
            if t < .4:
                self.assertEqual((len(d.route_calls),r.calls),(0,0))
        self.assertEqual(len(d.route_calls),1)
        self.assertEqual(r.calls,1)
        self.assertEqual({p['track_id'] for p in event['predictions']},{1})
        self.assertEqual({p['surface'] for p in event['predictions']},{'unknown'})
        self.assertEqual(event['selected_track_id'],1)

    def test_buffers_before_selection_without_ocr_and_only_selected_bus_decides(self):
        d, r = RelativeDetector(), ColorRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,temporal=False,camera_check=False,
                                preselection_collection=True)
        image = np.zeros((200,300,3),np.uint8)
        image[:,150:,0] = 200
        for i,t in enumerate([0,.2,.4]):
            d.current = [bus(t),bus(0,2,150)]
            event = pipe.process(frame(i,t,image))
            if i < 2:
                self.assertIsNone(event['decision'])
                self.assertEqual(event['predictions'],[])
                self.assertEqual(r.calls,0)
                self.assertEqual(set(pipe.raw_frames),{1,2})
                self.assertEqual(d.route_calls,[])
        self.assertEqual(event['selected_track_id'],1)
        self.assertEqual(event['decision']['state'],'recognized_single')
        self.assertEqual({p['track_id'] for p in event['predictions']},{1})
        self.assertEqual({p['surface'] for p in event['predictions']},{'unknown'})
        self.assertEqual(set(pipe.matcher.tracks),{1})

    def test_preselection_stays_inside_configured_entry_area(self):
        d,r = RelativeDetector(),FakeRecognizer()
        profile = ApproachProfile(entry_polygon=((0,0),(.4,0),(.4,1),(0,1)), direction=(0,1))
        pipe = ApproachPipeline(d,r,'606',profile,temporal=False,camera_check=False,
                                preselection_collection=True)
        d.current = [bus(0),bus(0,2,150)]
        event = pipe.process(frame(0,0))
        self.assertEqual(event['predictions'],[])
        self.assertEqual(d.route_calls,[])
        self.assertEqual(set(pipe.raw_frames),{1})
        self.assertEqual(r.calls,0)

    def test_first_selection_keeps_only_own_temporal_buffer(self):
        d,r = RelativeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False,preselection_collection=True)
        for i,t in enumerate([0,.2,.4]):
            d.current = [bus(t),bus(0,2,150)]
            event = pipe.process(frame(i,t))
            if i < 2:
                self.assertEqual(set(pipe.raw_frames),{1,2})
        self.assertEqual(event['selected_track_id'],1)
        self.assertEqual({display['track_id'] for display in pipe.displays.values()},{1})
        self.assertEqual([item['source_frame_index'] for display in pipe.displays.values()
                          for item in display['buffer'].frames],[2])
        self.assertEqual(set(pipe.sources),{(1,0),(1,1),(1,2)})
        self.assertEqual(set(pipe.raw_frames),{1})

    def test_preselection_rgb_ring_has_time_and_count_bounds(self):
        d,r = RelativeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,detection_interval=1,camera_check=False,
                                preselection_collection=True)
        d.current = [bus(0)]
        for i in range(20):
            item = frame(i,i/60)
            pipe.process(item)
            ring = pipe.raw_frames[1]
            self.assertLessEqual(len(ring),5)
            self.assertLessEqual(item['timestamp_s']-ring[0]['frame']['timestamp_s'],.4+1e-6)
        self.assertEqual(r.calls,0)
        self.assertEqual(d.route_calls,[])

    def test_60fps_detection_keyframe_enables_real_temporal_crop(self):
        canonical = np.zeros((110,150,3),np.uint8)
        cv2.putText(canonical,'606',(40,48),cv2.FONT_HERSHEY_SIMPLEX,1.1,
                    (255,255,255),2,cv2.LINE_AA)
        for visible in (True, False):
            d,r = RelativeTextDetector(),FakeRecognizer()
            pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False,preselection_collection=True)
            for i in range(25):
                t = i/60
                d.time = t
                bb = d.buses(None)[0]['bus_box']
                image = np.zeros((200,300,3),np.uint8)
                body = canonical if visible and i < 24 else np.zeros_like(canonical)
                image[bb[1]:bb[3],bb[0]:bb[2]] = cv2.resize(
                    body,(bb[2]-bb[0],bb[3]-bb[1]))
                event = pipe.process(frame(i,t,image))
                if i == 23:
                    ring = pipe.raw_frames[1]
                    self.assertLessEqual(len(ring),5)
                    self.assertIn(12,[item['frame']['source_frame_index'] for item in ring])
                    self.assertEqual(d.route_calls,[])
            self.assertEqual(event['selected_track_id'],1)
            if visible:
                self.assertEqual(event['decision']['state'],'recognized_single')
                self.assertEqual(len(event['predictions']),1)
                self.assertLess(event['predictions'][0]['source_frame_index'],24)
                self.assertLessEqual(.4-event['predictions'][0]['timestamp_s'],.25+1e-6)
                self.assertIn('retrospective_observed_frame_selected',
                              {q['reason'] for q in event['quality']})
            else:
                self.assertEqual(event['predictions'],[])
                self.assertEqual(r.calls,0)
                self.assertEqual(event['decision']['state'],'hold')

    def test_camera_motion_clears_preselection_evidence(self):
        d,r = RelativeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,temporal=False,preselection_collection=True)
        with patch('bus_route.service.camera_stable',side_effect=[True,True,False,True]):
            for i,t in enumerate([0,.2,.4]):
                d.current = [bus(t)]
                event = pipe.process(frame(i,t))
            self.assertIsNone(event['decision'])
            self.assertFalse(pipe.displays)
            self.assertFalse(pipe.sources)
            self.assertFalse(pipe.matcher.tracks)
            d.current = [bus(.6)]
            event = pipe.process(frame(3,.6))
        self.assertEqual(event['predictions'],[])
        self.assertFalse(pipe.displays)
        self.assertEqual({item['frame']['source_frame_index'] for item in pipe.raw_frames[1]},{3})

    def test_preselection_track_jump_and_overlap_discard_old_evidence(self):
        d,r = RelativeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,temporal=False,camera_check=False,
                                preselection_collection=True)
        for i,t in enumerate([0,.2]):
            d.current = [bus(t)]
            pipe.process(frame(i,t))
        d.current = [bus(.4,offset=150)]
        event = pipe.process(frame(2,.4))
        self.assertEqual(event['predictions'],[])
        self.assertEqual({item['frame']['source_frame_index'] for item in pipe.raw_frames[1]},{2})
        d.current = [bus(.6,offset=150),bus(.6,2,154)]
        event = pipe.process(frame(3,.6))
        self.assertEqual(event['predictions'],[])
        self.assertFalse(pipe.displays)
        self.assertFalse(pipe.matcher.tracks)

    def test_old_preselection_frame_cannot_become_selection_decision(self):
        d,r = RelativeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False,preselection_collection=True)
        old = {**frame(0,0), 'crop':np.zeros((20,40,3),np.uint8),
               'box':(15,18,50,36), 'relative_stroke_coverage':1.}
        with patch.object(DisplayBuffer,'best',return_value=(old,'test_old_source')):
            for i,t in enumerate([0,.2,.4]):
                d.current = [bus(t)]
                event = pipe.process(frame(i,t))
        self.assertEqual(event['selected_track_id'],1)
        self.assertEqual(event['predictions'],[])
        self.assertEqual(event['decision']['state'],'hold')

    def test_recent_preselection_frame_survives_one_selected_detector_miss(self):
        d,r = OneMissDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False,preselection_collection=True)
        recent = {**frame(1,.2), 'crop':np.zeros((20,40,3),np.uint8),
                  'box':(15,18,50,36), 'relative_stroke_coverage':1.}
        with patch('bus_route.service.propagate_boxes',side_effect=lambda previous, current, boxes: boxes), \
             patch.object(DisplayBuffer,'best',return_value=(recent,'recent_buffered_source')) as best:
            for i,t in enumerate([0,.2,.4]):
                d.time = t
                d.current = [bus(t)]
                event = pipe.process(frame(i,t))
                if i < 2:
                    self.assertEqual(r.calls,0)
                    self.assertEqual(best.call_count,0)
        self.assertEqual(event['selected_track_id'],1)
        self.assertEqual(event['decision']['state'],'recognized_single')
        self.assertEqual(len(event['predictions']),1)
        self.assertEqual(event['predictions'][0]['source_frame_index'],1)
        self.assertEqual(event['predictions'][0]['timestamp_s'],.2)
        self.assertEqual(event['predictions'][0]['available_at_s'],.4)

    def test_filtered_current_boxes_still_allow_one_fallback_scan(self):
        d,r = FilteredCurrentDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False,preselection_collection=True)
        recent = {**frame(1,.2), 'crop':np.zeros((20,40,3),np.uint8),
                  'box':(15,18,50,36), 'relative_stroke_coverage':1.}
        with patch.object(DisplayBuffer,'best',return_value=(recent,'recent_buffered_source')):
            for i,t in enumerate([0,.2,.4]):
                d.current = [bus(t)]
                event = pipe.process(frame(i,t))
        self.assertEqual(len(d.route_calls),2)
        self.assertIn('retrospective_route_scan_used',{q['reason'] for q in event['quality']})
        self.assertEqual(event['decision']['state'],'recognized_single')

    def test_retrospective_scan_respects_realtime_budget(self):
        d,r = OneMissDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False,preselection_collection=True)
        for i,t in enumerate([0,.2,.4]):
            d.current = [bus(t)]
            item = frame(i,t)
            if i == 2:
                item['wall_deadline_monotonic'] = time.perf_counter()+.1
            event = pipe.process(item)
        self.assertEqual(len(d.route_calls),1)
        self.assertEqual(event['predictions'],[])
        self.assertIn('retrospective_route_scan_skipped_budget',
                      {q['reason'] for q in event['quality']})

    def test_budget_skipped_fallback_retries_with_recent_same_track_source(self):
        d,r = OneMissDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False,preselection_collection=True)
        recent = {**frame(1,.2), 'crop':np.zeros((20,40,3),np.uint8),
                  'box':(15,18,50,36), 'relative_stroke_coverage':1.}
        with patch.object(DisplayBuffer,'best',return_value=(recent,'recent_buffered_source')):
            for i,t in enumerate([0,.2,.4,.433333]):
                d.current = [bus(t)]
                item = frame(i,t)
                if i == 2:
                    item['wall_deadline_monotonic'] = time.perf_counter()+.1
                elif i == 3:
                    item['wall_deadline_monotonic'] = time.perf_counter()+1
                event = pipe.process(item)
                if i == 2:
                    self.assertEqual(set(pipe.raw_frames),{1})
                    self.assertEqual(len(d.route_calls),1)
        self.assertEqual(len(d.route_calls),2)
        self.assertEqual(event['decision']['state'],'recognized_single')
        self.assertEqual(event['predictions'][0]['source_frame_index'],1)

    def test_budget_skipped_ocr_seeding_retries_without_touching_normal_buffer(self):
        d,r = RelativeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False,preselection_collection=True)
        for i,t in enumerate([0,.2,.4]):
            d.current = [bus(t)]
            item = frame(i,t)
            if i == 2:
                item['wall_deadline_monotonic'] = time.perf_counter()+.05
            event = pipe.process(item)
        self.assertIn('retrospective_ocr_skipped_budget',
                      {q['reason'] for q in event['quality']})
        self.assertEqual(set(pipe.raw_frames),{1})
        self.assertEqual([item['source_frame_index'] for display in pipe.displays.values()
                          for item in display['buffer'].frames],[2])
        recent = {**frame(1,.2), 'crop':np.zeros((20,40,3),np.uint8),
                  'box':(15,18,50,36), 'relative_stroke_coverage':1.}
        item = frame(3,.433333)
        item['wall_deadline_monotonic'] = time.perf_counter()+1
        with patch('bus_route.service.propagate_boxes',side_effect=lambda previous,current,boxes: boxes), \
             patch.object(DisplayBuffer,'best',side_effect=[(None,'temporal_warmup'),
                                                           (recent,'recent_buffered_source')]):
            event = pipe.process(item)
        self.assertEqual(event['decision']['state'],'recognized_single')
        self.assertEqual(event['predictions'][0]['source_frame_index'],1)
        self.assertFalse(pipe.raw_frames)

    def test_lost_or_changed_bus_clears_number_and_led_evidence(self):
        d,r = FakeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,temporal=False,camera_check=False)
        for i,t in enumerate([0,.2,.4]):
            d.time = t
            pipe.process(frame(i,t))
        self.assertTrue(pipe.matcher.tracks)
        d.current = [bus(.6,3)]
        event = pipe.process(frame(3,.6))
        self.assertEqual(event['predictions'],[])
        self.assertFalse(pipe.matcher.tracks)
        self.assertFalse(pipe.displays)
        self.assertFalse(pipe.sources)

    def test_no_profile_means_no_ocr_calls(self):
        d,r = FakeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',camera_check=False)
        for i in range(4):
            d.time = i*.2
            pipe.process(frame(i,i*.2))
        self.assertEqual((d.route_calls,r.calls),([],0))

    def test_same_source_seen_via_another_display_cannot_add_support(self):
        d,r = FakeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,camera_check=False)
        best = {**frame(2,.4),'crop':np.zeros((20,40,3),np.uint8),
                'box':(15,18,50,36),'relative_stroke_coverage':1.}
        with patch.object(DisplayBuffer,'best',return_value=(best,'test_source')):
            for i,t in enumerate([0,.2,.4,.6]):
                d.time = t
                event = pipe.process(frame(i,t))
        self.assertTrue(event['predictions'])
        self.assertFalse(event['predictions'][0]['eligible'])
        self.assertEqual(event['predictions'][0]['rejection_reason'],'source_frame_already_counted_on_this_bus')
        self.assertNotEqual(event['decision']['state'],'recognized_single')

    def test_overlapping_display_never_enters_ocr(self):
        d,r = FakeDetector(),FakeRecognizer()
        pipe = ApproachPipeline(d,r,'606',PROFILE,temporal=False,camera_check=False)
        # A small foreground bus covers the display, but bus-level IoU is low.
        for i,t in enumerate([0,.2,.4]):
            d.current = [bus(t),{'track_id':2,'bus_box':(18,18,48,36)}]
            event = pipe.process(frame(i,t))
        self.assertEqual(event['predictions'],[])
        self.assertEqual(r.calls,0)


class NativeVideoTests(unittest.TestCase):
    def test_variable_frame_timestamps_preserved_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            writer = video_writer(path/'source.mp4',(64,48),5)
            for i in range(6):
                writer.send(np.full((48,64,3),i*30,np.uint8).tobytes())
            writer.close()
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-i',str(path/'source.mp4'),
                            '-vf',"select='not(eq(n,2))'",'-vsync','0',str(path/'vfr.mp4')],check=True)
            frames = list(native_frames(path/'vfr.mp4'))
            self.assertEqual(len(frames),5)
            np.testing.assert_allclose([f['timestamp_s'] for f in frames],[0,.2,.6,.8,1],atol=1e-5)
            self.assertEqual([f['source_frame_index'] for f in frames],list(range(5)))
            self.assertTrue(all(f['rgb'].shape == (48,64,3) for f in frames))

    def test_native_rate_cap_selects_real_source_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.mp4'
            writer = video_writer(path,(64,48),20)
            for i in range(10):
                writer.send(np.full((48,64,3),i*20,np.uint8).tobytes())
            writer.close()
            frames = list(native_frames(path,max_fps=5))
            self.assertEqual([f['source_frame_index'] for f in frames],[0,4,8])
            np.testing.assert_allclose([f['timestamp_s'] for f in frames],[0,.2,.4],atol=1e-5)


if __name__ == '__main__':
    unittest.main()
