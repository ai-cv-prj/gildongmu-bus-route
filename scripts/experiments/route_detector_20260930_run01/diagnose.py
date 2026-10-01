"""Frozen manual regions: lossless original-video OCR, never production accuracy."""
import json,hashlib,time
from dataclasses import asdict
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from bus_route.video import sample_frames
from bus_route.recognizer import PARSeqRecognizer
from bus_route.target import exact_route,token_quality
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/route_detector_20260930_run01'
labels=json.loads((O/'manual_labels.json').read_text());records={r['key']:r for r in json.loads((O/'proposals.json').read_text())}
(O/'manual_crops').mkdir(exist_ok=False);(O/'manual_contexts').mkdir(exist_ok=False)
crops=[]
for l in labels:
 r=records[l['key']];reader=sample_frames(ROOT/r['video_path'],5);next(reader)
 try:
  for t,arr in reader:
   if abs(t-r['timestamp_s'])<1e-6:break
  else:raise RuntimeError(l)
 finally:reader.close()
 im=Image.fromarray(arr);b=l['bbox'];h=b[3]-b[1];p=max(2,round(.1*h))
 context=[max(0,b[0]-max(70,b[2]-b[0])),max(0,b[1]-max(70,2*h)),min(im.width,b[2]+max(70,b[2]-b[0])),min(im.height,b[3]+max(70,2*h))]
 im.crop(context).save(O/'manual_contexts'/f"{l['key']}.png")
 variants={'exact':im.crop(b),'pad10':im.crop([max(0,b[0]-p),max(0,b[1]-p),min(im.width,b[2]+p),min(im.height,b[3]+p)])}
 if l.get('quad'):
  q=np.float32(l['quad']);w=round(max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])));h=round(max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1])))
  dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]);variants['rectified']=Image.fromarray(cv2.warpPerspective(arr,cv2.getPerspectiveTransform(q,dst),(w,h)))
 for v,c in variants.items():
  path=O/'manual_crops'/f"{l['key']}_{v}.png";c.save(path);crops.append(dict(**l,target=r['target'],variant=v,path=str(path.relative_to(O)),size=list(c.size)))
 print('extracted',l['key'],flush=True)
recognizer=PARSeqRecognizer();out=[]
for start in range(0,len(crops),16):
 batch=crops[start:start+16];preds=recognizer.predict_batch([Image.open(O/c['path']) for c in batch])
 for c,p in zip(batch,preds):out.append(dict(**c,prediction=asdict(p),quality=token_quality(p.token_scores),matches_visible=exact_route(p.text,c['visible_text']),matches_target=exact_route(p.text,c['target'])))
result=dict(label_sha256=hashlib.sha256((O/'manual_labels.json').read_bytes()).hexdigest(),model=recognizer.metadata(),results=out)
(O/'ocr_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for r in out:print(r['video_index'],r['variant'],r['visible_text'],r['prediction']['text'],round(r['quality'],3),r['matches_visible'],flush=True)
