import json,shutil,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01';D=O/'dataset_reviewed';shutil.copytree(O/'dataset',D);ss=json.loads((D/'manifest.json').read_text())
for s in ss:
 i=s['video_index']
 if s['kind']=='bus':
  if i==11:s['boxes'][-1]=[394,305,434,368]
  if i==13:s['boxes'][-1]=[149,259,184,302]
  if i==49:s['boxes'][-1]=[414,783,470,834]
  if i in [30,132]:s['boxes'].pop()
 text=''
 for a,b,c,d in s['boxes']:text+=f"0 {(a+c)/2/s['width']:.8f} {(b+d)/2/s['height']:.8f} {(c-a)/s['width']:.8f} {(d-b)/s['height']:.8f}\n"
 (D/'labels'/s['split']/f"{s['key']}.txt").write_text(text);s['sha256']=hashlib.sha256((D/s['image']).read_bytes()).hexdigest()
for a in ss:
 for b in ss:
  if a['split']!=b['split']:assert Path(a['video_path']).name[:8]!=Path(b['video_path']).name[:8] and a['sha256']!=b['sha256']
(D/'manifest.json').write_text(json.dumps(ss,ensure_ascii=False,indent=2));(D/'data.yaml').write_text(f'path: {D}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: route_display\n')
protocol=dict(seed=20260930,epochs=100,imgsz=960,batch=8,freeze=0,lr0=.0003,initial_weights='output/route_detector_20260930_run01/train_run/weights/best.pt',conf=.25,iou=.5,checkpoint='validation mAP50-95',selection='fixed one run, no test tuning',test_scope='same development holdout dates as previous experiment; not a new independent dataset',manifest_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),adoption_gate='Do not replace production if normal PRINT/LED or hard-negative controls regress. If candidate merits pipeline integration, full143+negative offline then realtime required.',counts=dict(Counter(s['split'] for s in ss)))
(O/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));print(protocol)
