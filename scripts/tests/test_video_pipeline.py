"""영상 시간축·좌표·트랙별 집계·미검출 상태 회귀 검사."""
import importlib.util
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from bus_route.detector import clip_box
from bus_route.recognizer import Prediction
from bus_route.video import sample_frames, video_writer

spec = importlib.util.spec_from_file_location('infer_video', Path(__file__).parents[1]/'infer_video.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Detector:
    def reset(self):
        self.index = -1
    def sync(self):
        pass
    def buses(self, rgb):
        self.index += 1
        return [] if self.index == 0 else [{'track_id':1,'bus_box':(0,0,64,48),'bus_score':.9}]
    def text_regions(self, rgb, box):
        return (0,0,64,20), [] if self.index == 1 else [(5,5,30,15)]


class Recognizer:
    def predict(self, image):
        assert image.size == (25,10)
        return Prediction('03A', [.8], 1.)


class VideoTests(unittest.TestCase):
    def test_clip_original_coordinates(self):
        json.dumps(clip_box([1,2,3,4],100,80))
        self.assertEqual(clip_box([-5.2,3.9,109.2,200],100,80),(0,3,100,80))

    def test_repeat_is_per_frame_and_track(self):
        rows = [dict(track_id=1,timestamp_s=0,text='03')]*4
        rows += [dict(track_id=2,timestamp_s=i,text='03') for i in [0,.2,.4]]
        rows += [dict(track_id=None,timestamp_s=0,text='03')]
        result = module.summarize_tracks(rows)
        self.assertEqual(result[0]['text_frame_counts'], {'03':1})
        self.assertEqual(result[0]['repeated_texts'], [])
        self.assertEqual(result[1]['repeated_texts'], ['03'])

    def test_video_sampling_and_no_stale_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'input.mp4'
            writer = video_writer(path,(64,48),10)
            for i in range(10):
                writer.send(np.full((48,64,3),i*20,np.uint8).tobytes())
            writer.close()
            result = module.analyze(path,root/'result',Detector(),Recognizer(),5)
            self.assertEqual(result['sample_count'],5)
            self.assertEqual(result['status_counts'],{'no_bus':1,'no_text_candidate':1,'ocr':3})
            self.assertEqual(result['tracks'][0]['text_frame_counts'],{'03A':3})
            reader = sample_frames(root/'result/annotated.mp4',5)
            meta = next(reader)
            frames = list(reader)
            self.assertEqual(len(frames),5)
            self.assertAlmostEqual(frames[-1][0],.8)
            self.assertAlmostEqual(meta['duration'],1.,places=1)


if __name__ == '__main__':
    unittest.main()
