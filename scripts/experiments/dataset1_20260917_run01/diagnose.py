"""기존 중간 프레임/수동 crop과 자동 검출을 같은 시점에서 비교한다."""
import argparse
import json
from pathlib import Path
import time
from PIL import Image
import numpy as np
import torch
from bus_route.detector import BusTextDetector, ROOT
from bus_route.recognizer import PARSeqRecognizer


def coverage(gt, box):
    a,b,c,d = gt
    x,y,z,w = box
    return max(0,min(c,z)-max(a,x))*max(0,min(d,w)-max(b,y))/((c-a)*(d-b))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('기존 결과를 덮어쓰지 않습니다.')
    torch.set_num_threads(4)
    detector=BusTextDetector('cuda')
    recognizer=PARSeqRecognizer('cuda')
    rows=json.loads((ROOT/'output/dataset1_20260916_run01/review.json').read_text())
    results=[]
    for row in rows:
        if row['index'] not in [1,2,26,42,50]: continue
        image=Image.open(row['frame_path']).convert('RGB')
        rgb=np.array(image)
        detector.reset()
        buses=detector.buses(rgb)
        candidates=[]
        searches=[]
        for bus in buses:
            search,boxes=detector.text_regions(rgb,bus['bus_box'])
            searches.append(search)
            for box in boxes:
                candidates.append({'box':box,'gt_coverage':coverage(row['bbox_xyxy'],box),
                                   'text':recognizer.predict(image.crop(box)).text})
        manual=recognizer.predict(Image.open(ROOT/row['crop_path'])).text
        expected='271' if row['index']==50 else row['route_id_gt']
        results.append({'sample_id':row['sample_id'],'timestamp_s':row['timestamp_s'],
            'expected_route_hint':expected,'manual_box':row['bbox_xyxy'],'manual_prediction':manual,
            'bus_boxes':[b['bus_box'] for b in buses], 'search_boxes':searches,
            'max_bus_gt_coverage':max([coverage(row['bbox_xyxy'],b['bus_box']) for b in buses],default=0),
            'max_search_gt_coverage':max([coverage(row['bbox_xyxy'],b) for b in searches],default=0),
            'candidates':candidates,
            'note':'coverage는 기존 수동 사각형의 포함 면적 비율이며 IoU/검출 정확도가 아님'})
        print(row['sample_id'],manual,'auto=',candidates,flush=True)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
