import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from types import SimpleNamespace
from bus_route.detector import BusTextDetector
from bus_route.recognizer import Prediction
from bus_route.video import video_writer,latest_frames

sys.path.insert(0,str(Path(__file__).parents[1]))
import infer_target_video as runner


class Detector:
    def reset(self):pass
    def sync(self):pass
    def buses(self,rgb):return [{'track_id':1,'bus_box':(0,0,64,48),'bus_score':.95}]
    def text_regions(self,rgb,box):return (0,0,64,18),[]
    def regions_in_box(self,rgb,box,canvas_size=None):return [(4,24,40,36)]


class Recognizer:
    def predict_batch(self,images):return [Prediction('07',[.99,.99,.99],1.) for _ in images]


class TargetVideoTests(unittest.TestCase):
    def test_fallback_evidence_gate_preserves_b_and_rejects_weak_craft_rival(self):
        rows=[
            {'region_source':'craft','text':'1000','token_scores':[.9376]*5,
             'eligible':True,'rejection_reason':''},
            {'region_source':'route_display','text':'1200','token_scores':[.9977]*5,
             'eligible':True,'rejection_reason':''},
            {'region_source':'craft','text':'1200','token_scores':[.9913]*5,
             'eligible':True,'rejection_reason':''},
            {'region_source':'craft','text':'GBUS','token_scores':[.999]*5,
             'eligible':True,'rejection_reason':''},
            {'region_source':'craft','text':'7737','token_scores':[.999]*5,
             'eligible':False,'rejection_reason':'small_clipped_or_ambiguous_bus'},
        ]
        runner.gate_fallback_evidence(rows)
        self.assertEqual([row['eligible'] for row in rows],[False,True,True,False,False])
        self.assertEqual(rows[0]['rejection_reason'],'fallback_low_token_score')
        self.assertEqual(rows[3]['rejection_reason'],'fallback_nonroute_text')
        self.assertEqual(rows[4]['rejection_reason'],'small_clipped_or_ambiguous_bus')
        matcher=runner.TargetMatcher('1200',single_score=.98)
        self.assertEqual(matcher.update(1,2.6,[rows[0]],(10,10,100,100))['state'],'unverified')
        self.assertEqual(matcher.update(1,2.8,[rows[1]],(10,10,100,100))['state'],'recognized_single')

    def test_route_detector_uses_craft_only_when_b_finds_no_regions(self):
        module_path=runner.ROOT/'scripts/experiments/bus_roi_finetune_20260930_run01/run_videos.py'
        spec=importlib.util.spec_from_file_location('route_experiment_for_test',module_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        detector=object.__new__(module.RouteDetector)
        detector.craft_fallback_on_empty=True
        detector.last_region_source='route_display'
        rgb=np.zeros((100,100,3),np.uint8)
        bus=(10,10,90,90)
        class RouteModel:
            def __init__(self,boxes):self.boxes=boxes
            def predict(self,*args,**kwargs):
                return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor(self.boxes)))]
        with patch.object(BusTextDetector,'text_regions',return_value=((10,10,90,38),[(20,20,40,30)])) as craft_upper, \
             patch.object(BusTextDetector,'regions_in_box',return_value=[(20,40,40,50)]) as craft_body:
            detector.route=RouteModel([])
            self.assertEqual(detector.text_regions(rgb,bus),((10,10,90,38),[(20,20,40,30)]))
            self.assertEqual(detector.last_region_source,'craft')
            self.assertEqual(detector.regions_in_box(rgb,(10,30,90,90)),[(20,40,40,50)])
            self.assertEqual((craft_upper.call_count,craft_body.call_count),(1,1))
            detector.route=RouteModel([[20,20,40,30]])
            self.assertEqual(detector.text_regions(rgb,bus),(bus,[(30,30,50,40)]))
            self.assertEqual(detector.last_region_source,'route_display')
            self.assertEqual(detector.regions_in_box(rgb,(10,30,90,90)),[])
            self.assertEqual((craft_upper.call_count,craft_body.call_count),(1,1))

    def test_intact_body_crop_replaces_clipped_upper_duplicate(self):
        rgb=np.zeros((200,200,3),np.uint8);buses=[{'bus_box':(10,10,190,190)}]
        items=[{'text_box':(30,51,90,73),'search_zone':'upper','search_box':(10,10,190,73)},
               {'text_box':(30,51,90,81),'search_zone':'body','search_box':(10,45,190,190)}]
        chosen,count,valid,truncated=runner.select_candidates(rgb,items,0,buses,24)
        self.assertEqual((count,valid,truncated),(1,1,False))
        self.assertEqual(chosen[0]['text_box'],(30,51,90,81))
        self.assertTrue(chosen[0]['eligible'])

    def test_rejected_regions_do_not_exhaust_valid_candidate_budget(self):
        rgb=np.zeros((200,200,3),np.uint8);buses=[{'bus_box':(10,10,190,190)}]
        valid={'text_box':(30,40,60,55),'search_zone':'upper','search_box':(10,10,190,100)}
        clipped={'text_box':(10,20,180,80),'search_zone':'upper','search_box':(10,10,190,100)}
        chosen,count,n,truncated=runner.select_candidates(rgb,[clipped,valid],0,buses,1)
        self.assertEqual(chosen[0]['text_box'],valid['text_box'])
        self.assertEqual((count,n,truncated),(2,1,False))
        other={**valid,'text_box':(80,40,120,55)}
        chosen,count,n,truncated=runner.select_candidates(rgb,[clipped,valid,other],0,buses,1)
        self.assertTrue(truncated)  # 유효 경쟁 번호를 보지 못하면 확정 금지.

    def source(self,path,fps,count):
        w=video_writer(path,(64,48),fps)
        for i in range(count):w.send(np.full((48,64,3),50+i,np.uint8).tobytes())
        w.close()

    def test_body_only_can_support_target_without_prefix(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'output') as tmp:
            path=Path(tmp);self.source(path/'input.mp4',5,8)
            summary=runner.analyze(path/'input.mp4',path/'result',Detector(),Recognizer(),5,'마포07',body_interval=.2)
            self.assertEqual(summary['target_match_samples'],8)
            self.assertEqual(summary['body_match_samples'],8)
            self.assertEqual(summary['first_matched_candidate_s'],.4)
            self.assertEqual(summary['first_recognized_s'],0.)
            self.assertEqual(summary['first_single_s'],0.)

    def test_phone_context_only_removes_raw_route_evidence(self):
        import csv
        import json
        class SmallTextDetector(Detector):
            def regions_in_box(self,rgb,box,canvas_size=None):
                return [(20,65,38,71)]
            def buses(self,rgb):
                return [{'track_id':1,'bus_box':(0,0,64,120),'bus_score':.95}]
        class PhoneRecognizer:
            def predict_batch(self,images):
                return [Prediction('1577',[.99]*5,1.) if image.height==6
                        else Prediction('1577-1987',[.99]*10,1.) for image in images]
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'output') as tmp:
            path=Path(tmp)
            writer=video_writer(path/'input.mp4',(64,120),5)
            for _ in range(3):writer.send(np.full((120,64,3),80,np.uint8).tobytes())
            writer.close()
            summary=runner.analyze(path/'input.mp4',path/'result',SmallTextDetector(),PhoneRecognizer(),5,'1577')
            self.assertIsNone(summary['first_recognized_s'])
            with (path/'result/predictions.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(row['text'],'1577')
                self.assertEqual(row['eligible'],'False')
                self.assertEqual(row['rejection_reason'],'embedded_phone_number')
                self.assertEqual(len(json.loads(row['context_checks'])),2)

    def test_latest_queue_drops_old_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.mp4';self.source(path,20,10)
            reader=latest_frames(path,20);meta=next(reader);timestamps=[]
            for timestamp,frame in reader:
                timestamps.append(timestamp);time.sleep(.09)
            self.assertLess(len(timestamps),10)
            self.assertEqual(timestamps,sorted(set(timestamps)))
            self.assertAlmostEqual(timestamps[-1],.45)
            self.assertIsNotNone(meta['replay_origin'])

    def test_overlapping_bus_candidate_is_ambiguous(self):
        buses=[{'bus_box':(0,0,100,100)},{'bus_box':(20,0,120,100)}]
        self.assertFalse(runner.eligible_box((30,20,60,30),0,buses,(200,200,3)))
        self.assertTrue(runner.eligible_box((5,20,19,30),0,buses,(200,200,3)))

    def test_same_pixels_in_duplicate_bus_tracks_restore_one_read(self):
        rgb=np.zeros((200,200,3),np.uint8)
        buses=[{'bus_box':(20,10,140,180),'bus_score':.60},
               {'bus_box':(25,12,145,178),'bus_score':.70}]
        def item(index,box,score):
            return {'bus_index':index,'bus_box':buses[index]['bus_box'],
                    'text_box':box,'search_box':buses[index]['bus_box'],
                    'text':'604','token_scores':[score]*4,'eligible':False,
                    'rejection_reason':'small_clipped_or_ambiguous_bus',
                    'duplicate_bus_recovered':False}
        current=[item(0,(60,60,100,80),.9992),item(1,(61,60,101,80),.9993)]
        runner.recover_duplicate_bus_candidate(rgb,current,buses)
        self.assertEqual([p['eligible'] for p in current],[False,True])
        self.assertEqual([p['duplicate_bus_recovered'] for p in current],[False,True])

        # Two distinct overlapping buses, as in the reviewed 9360 vehicle ID.
        buses[1]['bus_box']=(42,10,162,180)  # IoU below .8
        blocked=[item(0,(60,60,100,80),.9992),item(1,(61,60,101,80),.9993)]
        runner.recover_duplicate_bus_candidate(rgb,blocked,buses)
        self.assertFalse(any(p['eligible'] for p in blocked))

        buses[1]['bus_box']=(25,12,145,178)
        narrow=[item(0,(60,60,76,80),.9992),item(1,(61,60,77,80),.9993)]
        runner.recover_duplicate_bus_candidate(rgb,narrow,buses)
        self.assertFalse(any(p['eligible'] for p in narrow))  # Density filter remains.

        third=buses+[{'bus_box':(50,50,120,120),'bus_score':.9}]
        ambiguous=[item(0,(60,60,100,80),.9992),item(1,(61,60,101,80),.9993)]
        runner.recover_duplicate_bus_candidate(rgb,ambiguous,third)
        self.assertFalse(any(p['eligible'] for p in ambiguous))

        # General CRAFT text cannot use the route-display-only duplicate veto.
        craft=[{**item(0,(60,60,100,80),.9992),'region_source':'craft'},
               {**item(1,(61,60,101,80),.9993),'region_source':'craft'}]
        runner.recover_duplicate_bus_candidate(rgb,craft,buses)
        self.assertFalse(any(p['eligible'] for p in craft))


if __name__=='__main__':unittest.main()
