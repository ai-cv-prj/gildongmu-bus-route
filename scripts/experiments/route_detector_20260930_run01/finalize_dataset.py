"""Apply second visual annotation review before training; preserve draft dataset."""
import json,shutil,hashlib
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/route_detector_20260930_run01';D=O/'dataset_reviewed';shutil.copytree(O/'dataset',D)
ss=json.loads((D/'manifest.json').read_text())
additional={71:[[40,73,67,95]],70:[[86,48,125,91]],124:[[33,64,61,84]],138:[[79,33,107,65]]}
for s in ss:
 if s['kind']=='positive' and s['video_index'] in additional:s['boxes']+=additional[s['video_index']]
 if s['key']=='v064_015':
  s['box_original']=[539,409,589,455];s['boxes']=[[50,25,100,71]]
 if s['key']=='v009_005':
  im=Image.open(D/s['image']);im.crop([0,32,im.width,im.height]).save(D/s['image']);s['roi'][1]+=32;s['height']-=32;s['boxes']=[[a,b-32,c,d-32] for a,b,c,d in s['boxes']]
 if s['key']=='neg_031':
  im=Image.open(D/s['image']);im.crop([0,0,65,im.height]).save(D/s['image']);s['roi'][2]=s['roi'][0]+65;s['width']=65
 text=''
 for a,b,c,d in s['boxes']:
  assert 0<=a<c<=s['width'] and 0<=b<d<=s['height'],s
  text+=f"0 {(a+c)/2/s['width']:.8f} {(b+d)/2/s['height']:.8f} {(c-a)/s['width']:.8f} {(d-b)/s['height']:.8f}\n"
 (D/'labels'/s['split']/f"{s['key']}.txt").write_text(text)
 s['review_status']='visual context checked, complete secondary route displays added'
 s['sha256']=hashlib.sha256((D/s['image']).read_bytes()).hexdigest()
(D/'manifest.json').write_text(json.dumps(ss,ensure_ascii=False,indent=2))
(D/'data.yaml').write_text(f'path: {D}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: route_display\n')
# Day separation prevents adjacent frames / same capture event crossing splits.
for a in ss:
 for b in ss:
  if a['split']!=b['split']:assert a['video_path']!=b['video_path'] and a['sha256']!=b['sha256']
protocol=dict(scope='oracle local-window localization pilot; full-frame deployment untested',seed=20260930,epochs=60,imgsz=640,batch=16,freeze=10,conf=.25,iou=.5,complete_coverage=.95,max_area_ratio=2.,train_days=['20260913','20260914','20260915','20260916','20260917'],val_days=['20260920'],test_days=['20260918','20260921'],checkpoint='best validation mAP50-95, no test tuning',manual_labels_sha256=hashlib.sha256((O/'manual_labels.json').read_bytes()).hexdigest(),dataset_manifest_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),limitations=['small sample','same routes and locations may repeat on different days','positive selection based on visible region','two heldout negative windows only','human-readable annotation uncertainty excluded, not video failures removed'])
(O/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
print('Reviewed dataset frozen',len(ss))
