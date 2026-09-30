# 파일명: freeze_annotations.py
# 설명: 모델 실행 전에 육안 가시성 라벨을 고정하고 평가 CSV를 생성한다.
import csv
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[3]/'output/dataset1_20260916_run01'
READABLE={1,2,3,4,6,7,10,14,16,17,18,24,26,28,30,31,32,33,39,41,43,46,48,49,51,52,53,54,55,58}
NO_ROUTE={35,60,61,62}
OCCLUDED={20}
CONFLICT={50}
with (OUT/'annotations.csv').open() as f: rows=list(csv.DictReader(f))
review=json.loads((OUT/'review.json').read_text())
by_id={r['index']:r for r in review}
for row in rows:
    i=int(row['sample_id'].split('_')[1])
    row['target_text_from_filename']=row['route_id_gt'].removeprefix('서대문')
    row['filename_label_conflict']='yes' if i in CONFLICT else 'no'
    row['filename_condition']='fault' if '불량' in row['video_id'] else 'glare' if '빛번짐' in row['video_id'] else 'untagged'
    row['visibility']='readable' if i in READABLE else 'no_route' if i in NO_ROUTE else 'occluded' if i in OCCLUDED else 'unverified' if i in CONFLICT else 'partial'
    row['visible_text_gt']=row['target_text_from_filename'] if i in READABLE else '271' if i in CONFLICT else ''
    row['gt_source']='filename + assistant manual crop review BEFORE inference; not independent blind ground truth'
    r=by_id[i]
    r.update(visibility=row['visibility'],visible_text_gt=row['visible_text_gt'],review_status='reviewed_before_inference')
    if i in CONFLICT:r['note']='filename route 272 conflicts with visible 271; excluded from scored accuracy'
    if i in NO_ROUTE:r['note']='selected LED area shows a message rather than an identifiable route number'
    if i==56:r['note']='strong reflection; first digit incomplete/uncertain'
with (OUT/'annotations.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
from collections import Counter
print(Counter(r['visibility'] for r in rows))
