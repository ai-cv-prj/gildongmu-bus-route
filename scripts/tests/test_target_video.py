import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
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


if __name__=='__main__':unittest.main()
