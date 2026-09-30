"""실제 파일명 예외 및 사후 집계에서 정답 치환/중복 집계를 막는다."""
import importlib.util
from pathlib import Path
import unittest


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


prepare=load('prepare_video_inputs')
summary=load('summarize_videos')


class ReportingTests(unittest.TestCase):
    def test_orientation_token_and_route_characters(self):
        for name,expected in [('20260921_갤럭시S25U_세로_273_LED.mp4','273'),
                              ('20260917_갤럭시quantum3_마포07_LED_불량.mp4','마포07'),
                              ('20260917_갤럭시quantum3_110A_PRINT.mp4','110A'),
                              ('20260918_갤럭시quantum3_M6628_LED.mp4','M6628')]:
            self.assertEqual(prepare.describe_video(prepare.ROOT/'data/video/dataset2'/name)['route_hint'],expected)

    def test_tags_and_invalid_route(self):
        self.assertEqual(prepare.describe_video(prepare.ROOT/'data/video/dataset2/20260918_iPhone15pro_M7731_빛반사.mov')['condition_tag'],'빛반사')
        with self.assertRaises(ValueError):
            prepare.describe_video(prepare.ROOT/'data/video/dataset2/20260921_phone_세로_LED.mp4')

    def test_exact_text_and_track_separation(self):
        rows=[dict(text='07',timestamp_s='0.0',track_id='1')]*3
        rows += [dict(text='07',timestamp_s='.2',track_id='2'),dict(text='07',timestamp_s='.4',track_id='')]
        result=summary.matching_summary(rows,'07')
        self.assertEqual(result['matching_sample_count'],3)
        self.assertEqual(result['max_matching_samples_same_track'],1)
        self.assertEqual(result['first_matching_candidate_s'],0.)
        self.assertEqual(summary.matching_summary(rows,'마포07')['matching_sample_count'],0)
        self.assertEqual(summary.matching_summary([dict(text='110A',timestamp_s='0',track_id='1')],'110B')['matching_sample_count'],0)
        self.assertEqual(summary.matching_summary([],'271')['first_matching_candidate_s'],'')


if __name__=='__main__':unittest.main()
