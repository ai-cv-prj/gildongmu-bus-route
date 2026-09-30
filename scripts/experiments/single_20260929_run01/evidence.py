"""Extract all newly recognized target videos and a declared diagnostic negative subset."""
import csv,json,collections
from pathlib import Path
from PIL import Image,ImageDraw
from bus_route.video import sample_frames
from bus_route.target import exact_route,token_quality,competing_route
R=Path(__file__).resolve().parents[3];O=R/'output/single_20260929_run01';B=R/'output/hold_fix_20260929_run01';D=O/'evidence';D.mkdir(exist_ok=True)
s=json.loads((O/'offline/summary.json').read_text());inputs=json.loads((B/'summary.json').read_text());labels={f'video_{i:02d}':x['target_route'] for i,x in enumerate(inputs,1)}
items=[{**r,'kind':'gain'} for r in s['gains']]
flags=[r for r in json.loads((O/'offline/alternative_flags.json').read_text()) if not r['baseline_also_passed']]
# Inspect every one-character confusion plus representative partial/plate/advert risks.
selected=[r for r in flags if competing_route(r['target'],labels[r['folder']])]
for folder,target in [('video_01','3400'),('video_04','153'),('video_07','27'),('video_11','2026'),('video_13','7284'),('video_72','100'),('video_97','마포11'),('video_102','70'),('video_128','10'),('video_138','1577'),('video_138','9350')]:
 selected.extend(r for r in flags if r['folder']==folder and r['target']==target and r not in selected)
items.extend({**r,'kind':'alternative'} for r in selected)
groups=collections.defaultdict(list);records=[]
for i,r in enumerate(items):
 with (B/r['folder']/'predictions.csv').open(encoding='utf-8-sig') as f:ps=list(csv.DictReader(f))
 p=max((p for p in ps if float(p['timestamp_s'])==r['first_s'] and p['eligible']=='True' and exact_route(p['text'],r['target']) and str(r['track_id'])==p['track_id']),key=lambda p:token_quality(json.loads(p['token_scores'])))
 row={**r,'id':i,'box':json.loads(p['text_box']),'ocr_text':p['text'],'score':token_quality(json.loads(p['token_scores'])),'crop':f'{i:02d}_crop.png','context':f'{i:02d}_context.jpg'};records.append(row);groups[r['video_path']].append(row)
for path,rs in groups.items():
 reader=sample_frames(R/path,5);next(reader)
 try:
  for t,rgb in reader:
   for r in rs:
    if abs(t-r['first_s'])>1e-6:continue
    im=Image.fromarray(rgb);im.crop(r['box']).save(D/r['crop']);ImageDraw.Draw(im).rectangle(r['box'],outline='red',width=5);im.thumbnail((640,900));im.save(D/r['context'],quality=95)
   if t>=max(r['first_s'] for r in rs):break
 finally:reader.close()
(D/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
for start in range(0,len(records),6):
 im=Image.new('RGB',(1440,1240),'#eee');draw=ImageDraw.Draw(im)
 for j,r in enumerate(records[start:start+6]):
  x=j%3*480;y=j//3*620;draw.text((x+5,y+5),f'{r["id"]} {r["kind"]} {r["folder"]} target={r["target"]} t={r["first_s"]}',fill='black');ctx=Image.open(D/r['context']);ctx.thumbnail((465,460));im.paste(ctx,(x+5,y+30));c=Image.open(D/r['crop']);scale=min(460/c.width,110/c.height,6);c=c.resize((max(1,round(c.width*scale)),max(1,round(c.height*scale))));im.paste(c,(x+5,y+505))
 im.save(D/f'sheet_{start//6:02d}.jpg',quality=95)
print('extracted',len(records))
