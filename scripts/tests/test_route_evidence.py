import unittest
import numpy as np
from bus_route.route_evidence import crop_rejection,text_rejection


class RouteEvidenceTests(unittest.TestCase):
    def test_shadowed_and_blurred_yellow_plate(self):
        for color in [[80,65,10],[255,225,40]]:
            rgb=np.zeros((100,100,3),dtype=np.uint8)
            rgb[80:90,30:70]=color
            self.assertEqual(crop_rejection(rgb,(30,80,70,90),(0,0,100,100),(0,25,100,100)),
                             'likely_registration_plate')

    def test_lower_yellow_plate_excluded_but_upper_yellow_led_not(self):
        rgb=np.zeros((100,100,3),dtype=np.uint8)
        rgb[80:90,30:70]=[240,200,30]
        rgb[80:90,40:45]=0
        self.assertEqual(crop_rejection(rgb,(30,80,70,90),(0,0,100,100),(0,25,100,100)),
                         'likely_registration_plate')
        rgb[10:20,30:70]=rgb[80:90,30:70]
        self.assertEqual(crop_rejection(rgb,(30,10,70,20),(0,0,100,100),(0,0,100,35)),'')

    def test_truncated_body_led_rejected_without_blocking_intact_print(self):
        rgb=np.zeros((100,100,3),dtype=np.uint8)
        self.assertEqual(crop_rejection(rgb,(20,25,50,35),(0,0,100,100),(0,25,100,100)),
                         'search_boundary_clipped')
        self.assertEqual(crop_rejection(rgb,(20,40,50,60),(0,0,100,100),(0,25,100,100)),'')

    def test_narrow_hangul_misread_but_two_digit_print_preserved(self):
        self.assertTrue(text_rejection('1010',(0,0,21,43)))
        self.assertTrue(text_rejection('100',(0,0,16,30)))
        self.assertFalse(text_rejection('11',(0,0,12,30)))
        self.assertFalse(text_rejection('110A',(0,0,67,43)))


class RecordedRegionTests(unittest.TestCase):
    def test_real_plates_and_side_print(self):
        import json
        from pathlib import Path
        from PIL import Image
        root = Path(__file__).parent/'fixtures/route_evidence'
        for case in json.loads((root/'cases.json').read_text()):
            with self.subTest(text=case['text'], video=case['video']):
                rgb = np.array(Image.open(root/f'{case["id"]:02d}.png'))
                self.assertEqual(crop_rejection(rgb,case['box'],case['bus_box'],case['search_box']),
                                 case['expected'])

    def test_phone_context_rejects_both_halves_but_never_repairs_route(self):
        from bus_route.route_evidence import context_rejection
        self.assertEqual(context_rejection('1577','1577-1987',[.99]*10),'embedded_phone_number')
        self.assertEqual(context_rejection('1987','1577-1987',[.99]*10),'embedded_phone_number')
        for text, context, scores in [('7011','7011',[.99]*5),
                                     ('701','7016',[.99]*5),
                                     ('7016','7017-1234',[.99]*10),
                                     ('1577','1577-1987',[.99]*4+[.5]+[.99]*5),
                                     ('1577','1577-1987',[]),
                                     ('1577','1577-1987',[.99]*9)]:
            with self.subTest(text=text, context=context, scores=scores):
                self.assertFalse(context_rejection(text,context,scores))

    def test_context_crosses_detector_box_without_adding_evidence(self):
        from bus_route.route_evidence import context_boxes
        p={'eligible':True,'text':'1577','token_scores':[.99]*5,
           'text_box':(835,280,856,291),'bus_box':(717,182,866,477)}
        boxes=context_boxes(p,(1920,1080,3))
        self.assertEqual(boxes,[(824,277,900,294),(791,277,867,294)])
        self.assertGreater(boxes[0][2],p['bus_box'][2])
        self.assertFalse(context_boxes({**p,'eligible':False},(1920,1080,3)))
        self.assertFalse(context_boxes({**p,'text':'M7731'},(1920,1080,3)))
        self.assertFalse(context_boxes({**p,'text':'07'},(1920,1080,3)))
