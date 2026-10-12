"""Collect diagnostic frames and region proposals; nothing is a reviewed label yet."""
import csv,json,re,base64,io,sys
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from PIL import Image,ImageDraw
from bs4 import BeautifulSoup
from bus_route.video import sample_frames
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/route_detector_20260930_run01';BASE=ROOT/'output/region_recovery_20260930_run01'
sys.path.insert(0,str(ROOT/'scripts/experiments/region_recovery_20260930_run01'))
from evaluate import load
from bus_route.target import exact_route,token_quality
(OUT/'frames').mkdir();(OUT/'sheets').mkdir();(OUT/'proposals').mkdir()
summaries=json.loads((BASE/'offline/summary.json').read_text());cards=BeautifulSoup((ROOT/'workbench/team_report.html').read_text(),'html.parser').select('.case-card')
records=[]
for i,(s,card) in enumerate(zip(summaries[:143],cards),1):
 failed=s['first_recognized_s'] is None
 caption=card.select_one('.zoomable').get('data-caption','')
 t=float(re.search(r'([0-9]+\.[0-9]+)초',caption)[1])
 if not failed:t=s['first_recognized_s']
 data=load(BASE/'offline'/f'video_{i:02}')
 ps=[p for p in data[1] if float(p['timestamp_s'])==t and exact_route(p['text'],s['target_route'])]
 box=max(ps,key=lambda p:token_quality(p['token_scores']))['text_box'] if ps else None
 # Try matching the exact previously published crop back to the original frame.
 encoded=card.select_one('.zoomable img')['src'].split(',',1)[1]
 template=np.array(Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB'))
 times=[t]
 if failed:
  last=float(data[0][-1]['timestamp_s'])
  for extra in [round(last*.35*5)/5,round(last*.75*5)/5]:
   if extra not in times:times.append(extra)
  if i==103:times.append(9.8)
  if i==135:times.append(1.2)
 reader=sample_frames(ROOT/s['video'],5);meta=next(reader)
 saved=[]
 try:
  for stamp,rgb in reader:
   if all(abs(stamp-v)>1e-6 for v in times):continue
   key=f'v{i:03}_{round(stamp*5):03}';image=Image.fromarray(rgb);image.save(OUT/'frames'/f'{key}.jpg',quality=98)
   frame=next(f for f in data[0] if abs(float(f['timestamp_s'])-stamp)<1e-6)
   frameps=[p for p in data[1] if abs(float(p['timestamp_s'])-stamp)<1e-6]
   proposal=box if abs(stamp-t)<1e-6 else None
   match_score=None
   if abs(stamp-t)<1e-6 and template.shape[0]<=rgb.shape[0] and template.shape[1]<=rgb.shape[1]:
    corr=cv2.matchTemplate(rgb,template,cv2.TM_CCOEFF_NORMED);_,match_score,_,pos=cv2.minMaxLoc(corr)
    if match_score>.995:proposal=[pos[0],pos[1],pos[0]+template.shape[1],pos[1]+template.shape[0]]
   item=dict(key=key,video_index=i,video_path=s['video'],target=s['target_route'],failed=failed,timestamp_s=stamp,frame_path=f'frames/{key}.jpg',width=image.width,height=image.height,proposal=proposal,template_match=match_score,baseline_cause=card.get('data-cause'),buses=frame['buses'],ocr=[{k:p[k] for k in ['text','text_box','eligible','rejection_reason','token_scores']} for p in frameps],review_status='unreviewed')
   records.append(item);saved.append(item)
   if proposal:
    image.crop(proposal).save(OUT/'proposals'/f'{key}.png')
   # Inspection coordinate system is the original image, scales recorded below.
   image.thumbnail((600,650));draw=ImageDraw.Draw(image);sx=image.width/rgb.shape[1];sy=image.height/rgb.shape[0]
   if proposal:draw.rectangle([proposal[0]*sx,proposal[1]*sy,proposal[2]*sx,proposal[3]*sy],outline='red',width=2)
   item['preview_size']=[image.width,image.height]
   image.save(OUT/'sheets'/f'{key}_context.jpg',quality=95)
   if stamp>=max(times)-1e-6:break
 finally:reader.close()
 if failed:
  sheet=Image.new('RGB',(600*len(saved),720),'#eee');draw=ImageDraw.Draw(sheet)
  for j,r in enumerate(saved):
   draw.text((j*600+5,5),f'{r["key"]} target={r["target"]} t={r["timestamp_s"]} original={r["width"]}x{r["height"]}',fill='black')
   im=Image.open(OUT/'sheets'/f'{r["key"]}_context.jpg');sheet.paste(im,(600*j,40))
  sheet.save(OUT/'sheets'/f'failure_{i:03}.jpg',quality=95)
 print(i,'failed' if failed else 'success',flush=True)
(OUT/'proposals.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
# Compact proposal crops for visual review, all held to unreviewed until signed off.
for start in range(0,143,24):
 sheet=Image.new('RGB',(1600,1200),'#eee');draw=ImageDraw.Draw(sheet)
 for j,i in enumerate(range(start+1,min(start+25,144))):
  r=next(x for x in records if x['video_index']==i)
  x=j%4*400;y=j//4*200
  draw.text((x+5,y+5),f'{i} target={r["target"]} failed={r["failed"]} {r["key"]}',fill='black')
  path=OUT/'proposals'/f'{r["key"]}.png'
  if path.exists():
   im=Image.open(path);scale=min(380/im.width,150/im.height,5);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))));sheet.paste(im,(x+5,y+35))
 sheet.save(OUT/'sheets'/f'proposals_{start:03}.jpg',quality=95)
