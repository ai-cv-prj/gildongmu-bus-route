import sys
from pathlib import Path
import unittest
import numpy as np
from bus_route.recognizer import Prediction
from bus_route.region_recovery import recovery_boxes, confirmation_box, is_extension
sys.path.insert(0,str(Path(__file__).parents[1]))
from infer_target_video import recover_regions


class RecoveryTests(unittest.TestCase):
    def source(self, text='42', reason=''):
        return dict(text=text,token_scores=[.999]*(len(text)+1),text_box=(100,90,154,120),
                    bus_box=(10,10,400,300),search_box=(10,10,400,160),bus_index=0,
                    eligible=not reason,rejection_reason=reason,context_checks=[])

    def run_recovery(self, text, confirm=None, reason='', rgb=None):
        source=self.source(reason=reason)
        class OCR:
            def predict_batch(self, images):
                return [Prediction(text if im.width<=99 else (confirm or text),[.999]*(len(text)+1),1.) for im in images]
        current=[source]
        recover_regions(np.zeros((320,420,3),np.uint8) if rgb is None else rgb,current,[{'bus_box':source['bus_box']}],OCR())
        return current

    def test_same_digits_two_larger_pixel_crops_required(self):
        ps=self.run_recovery('4215')
        self.assertEqual([p['text'] for p in ps if p['eligible']],['4215'])
        self.assertEqual(ps[0]['rejection_reason'],'superseded_by_full_region')
        self.assertEqual(ps[-1]['recovered_from']['text'],'42')

    def test_disagreeing_confirmation_keeps_original(self):
        ps=self.run_recovery('4215','4216')
        self.assertEqual(len(ps),1)
        self.assertTrue(ps[0]['eligible'])

    def test_unchanged_recognition_keeps_normal_route(self):
        ps=self.run_recovery('42')
        self.assertEqual(len(ps),1)
        self.assertTrue(ps[0]['eligible'])

    def test_original_digits_cannot_be_changed_or_guessed(self):
        for text in ['4315','M43','42-1500','104215','bad42']:
            with self.subTest(text=text):
                self.assertFalse(is_extension('42',text,[.999]*(len(text)+1),'right'))
                self.assertFalse(is_extension('42',text,[.999]*(len(text)+1),'left'))
        self.assertTrue(is_extension('4281','N4281',[.999]*6,'left'))
        self.assertFalse(is_extension('4281','N4281',[.97]*6,'left'))
        self.assertFalse(is_extension('42','4215',[.999]*4,'right'))

    def test_adjacent_numeric_line_cannot_become_a_prefix(self):
        self.assertFalse(is_extension('428','6428',[.999]*5,'left'))
        self.assertFalse(is_extension('42','4215',[float('nan')]*5,'right'))
        self.assertFalse(is_extension('42','4215',[float('inf')]*5,'right'))

    def test_right_confirmation_also_checks_missing_left_characters(self):
        p=self.source()
        side,box=next(x for x in recovery_boxes(p,(320,420,3)) if x[0]=='right')
        confirm=confirmation_box(box,side,p['text_box'],p['search_box'])
        self.assertLess(confirm[0],box[0])
        self.assertGreater(confirm[2],box[2])

    def test_rejected_plate_phone_boundary_never_seed_extensions(self):
        for reason in ['likely_registration_plate','embedded_phone_number','search_boundary_clipped','small_clipped_or_ambiguous_bus']:
            self.assertEqual(recovery_boxes(self.source(reason=reason),(320,420,3)),[])
            self.assertEqual(len(self.run_recovery('4215',reason=reason)),1)

    def test_extensions_stay_inside_search_region(self):
        p=self.source();p['search_box']=(95,80,175,130)
        self.assertEqual(recovery_boxes(p,(320,420,3)),[])

    def test_ambiguous_other_bus_cannot_supply_extended_evidence(self):
        p=self.source();ps=[p]
        class OCR:
            def predict_batch(self, images):return [Prediction('4215',[.999]*5,1.) for _ in images]
        recover_regions(np.zeros((320,420,3),np.uint8),ps,[{'bus_box':p['bus_box']},{'bus_box':(130,70,220,150)}],OCR())
        self.assertEqual(len(ps),1)
        self.assertTrue(ps[0]['eligible'])


if __name__=='__main__':unittest.main()
