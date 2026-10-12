"""Small localization pilot on reviewed local windows; not full-frame ground truth."""
import json,sys,re,hashlib
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
import numpy as np
from bus_route.target import exact_route,token_quality
from bus_route.video import sample_frames
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/route_detector_20260930_run01';BASE=ROOT/'output/region_recovery_20260930_run01'
sys.path.insert(0,str(ROOT/'scripts/experiments/region_recovery_20260930_run01'));from evaluate import load
records=json.loads((O/'proposals.json').read_text());manual={l['video_index']:l for l in json.loads((O/'manual_labels.json').read_text())}
D=O/'dataset';D.mkdir();samples=[]
def split(path):
 day=Path(path).name[:8]
 return 'test' if day in ['20260918','20260921'] else 'val' if day=='20260920' else 'train'
def window(b,w,h):
 bw=b[2]-b[0];bh=b[3]-b[1];return [max(0,b[0]-max(50,bw)),max(0,b[1]-max(40,2*bh)),min(w,b[2]+max(50,bw)),min(h,b[3]+max(40,2*bh))]
def save(key,path,t,im,b,kind,extra):
 spl=split(path);roi=window(b,im.width,im.height);crop=im.crop(roi)
 for p in ['images','labels']:(D/p/spl).mkdir(parents=True,exist_ok=True)
 crop.save(D/'images'/spl/f'{key}.png');boxes=[]
 if kind=='positive':boxes=[[b[0]-roi[0],b[1]-roi[1],b[2]-roi[0],b[3]-roi[1]]]
 text=''
 for x1,y1,x2,y2 in boxes:text+=f'0 {(x1+x2)/2/crop.width:.8f} {(y1+y2)/2/crop.height:.8f} {(x2-x1)/crop.width:.8f} {(y2-y1)/crop.height:.8f}\n'
 (D/'labels'/spl/f'{key}.txt').write_text(text)
 samples.append(dict(key=key,video_path=path,timestamp_s=t,split=spl,kind=kind,roi=roi,box_original=b,boxes=boxes,width=crop.width,height=crop.height,image=f'images/{spl}/{key}.png',**extra))
for i in range(1,144):
 rs=[r for r in records if r['video_index']==i];r=rs[0]
 if r['failed']:
  l=manual[i]
  if l['visibility']!='readable':continue
  r=next(x for x in rs if x['key']==l['key']);b=l['bbox'];text=l['visible_text'];origin='manual'
 else:
  _,ps,_=load(BASE/'offline'/f'video_{i:02}')
  ps=[p for p in ps if float(p['timestamp_s'])==r['timestamp_s'] and p['eligible'] and exact_route(p['text'],r['target'])]
  p=max(ps,key=lambda p:token_quality(p['token_scores']));b=p['text_box'];text=p['text'];origin='visually reviewed existing complete OCR region'
 save(r['key'],r['video_path'],r['timestamp_s'],Image.open(O/r['frame_path']),b,'positive',dict(video_index=i,baseline_failed=r['failed'],visible_text=text,label_origin=origin))
# Only reviewed non-route evidence. Do not label partial/damaged route numbers as background.
old=ROOT/'output/plate_filter_20260929_run01';reviews=json.loads((old/'evidence/review.json').read_text())
for rv in reviews:
 if rv['id'] not in [12,16,18,19,20,23,24,25,31,33]:continue
 reader=sample_frames(ROOT/rv['video_path'],5);next(reader)
 try:
  for t,arr in reader:
   if abs(t-rv['first_s'])<1e-6:break
  else:raise RuntimeError(rv)
 finally:reader.close()
 save(f"neg_{rv['id']:03}",rv['video_path'],t,Image.fromarray(arr),rv['box'],'negative',dict(video_index=int(rv['folder'].split('_')[1]),baseline_failed=None,visible_text=None,label_origin=rv['review_note']))
(D/'manifest.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2))
(D/'data.yaml').write_text(f'path: {D}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: route_display\n')
for start in range(0,len(samples),24):
 sheet=Image.new('RGB',(1200,180*6),'#eee');draw=ImageDraw.Draw(sheet)
 for j,s in enumerate(samples[start:start+24]):
  im=Image.open(D/s['image']);drawim=ImageDraw.Draw(im)
  for b in s['boxes']:drawim.rectangle(b,outline='red',width=2)
  im.thumbnail((296,150));x=j%4*300;y=j//4*180;sheet.paste(im,(x,y+25));draw.text((x+3,y+3),f"{s['key']} {s['split']} {s['visible_text']}",fill='black')
 sheet.save(O/'sheets'/f'dataset_{start:03}.jpg')
print(Counter((s['split'],s['kind']) for s in samples))
