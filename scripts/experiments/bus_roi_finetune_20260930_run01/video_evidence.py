"""Original RGB evidence for every gain/loss/newly passed alternative."""
import json,sys
from pathlib import Path
from collections import defaultdict
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01';BASE=ROOT/'output/region_recovery_20260930_run01'
sys.path.insert(0,str(ROOT/'scripts/experiments/plate_filter_20260929_run01'));from evaluate import load
from bus_route.target import exact_route,token_quality
from bus_route.video import sample_frames
E=O/'video_evidence';E.mkdir();rows=[];grouped=defaultdict(list)
for mode in ['offline','realtime']:
 e=json.loads((O/f'{mode}_comparison.json').read_text());cases=[dict(r,kind='gain',time=r['first_s']) for r in e['gains']]+[dict(r,kind='loss',time=r['old_first_s']) for r in e['losses']]+[dict(r,kind='new_alternative',time=r['first']['first_s']) for r in e['new_alternative_flags']]
 for c in cases:
  data=load((BASE if c['kind']=='loss' else O)/mode/c['folder']);ps=[p for p in data[1] if float(p['timestamp_s'])==c['time'] and p['eligible'] and exact_route(p['text'],c['target'])]
  if c['kind']=='new_alternative':ps=[p for p in ps if str(c['first']['track_id'])==p['track_id']]
  p=max(ps,key=lambda p:token_quality(p['token_scores']));r=dict(**c,id=len(rows),mode=mode,box=p['text_box'],bus_box=p['bus_box'],text=p['text'],quality=token_quality(p['token_scores']),verdict='needs_review');rows.append(r);grouped[c['video_path']].append(r)
for path,cases in grouped.items():
 reader=sample_frames(ROOT/path,5);next(reader)
 try:
  for t,rgb in reader:
   for r in cases:
    if abs(t-r['time'])>1e-6:continue
    im=Image.fromarray(rgb);im.crop(r['box']).save(E/f"{r['id']:03}_crop.png");draw=ImageDraw.Draw(im);draw.rectangle(r['bus_box'],outline='orange',width=3);draw.rectangle(r['box'],outline='red',width=3);a,b,c,d=r['box'];pad=max(c-a,d-b,50);zoom=im.crop([max(0,a-pad),max(0,b-pad),min(im.width,c+pad),min(im.height,d+pad)]);zoom.thumbnail((650,500));zoom.save(E/f"{r['id']:03}_detail.jpg",quality=95);im.thumbnail((600,800));im.save(E/f"{r['id']:03}_context.jpg",quality=95)
   if t>=max(r['time'] for r in cases):break
 finally:reader.close()
(E/'review_draft.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
for start in range(0,len(rows),12):
 sheet=Image.new('RGB',(1440,450*3),'#ddd');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(rows[start:start+12]):
  x=j%4*360;y=j//4*450;draw.text((x+3,y+3),f"{r['id']} {r['mode']} {r['kind']} {r['folder']} {r['target']}",fill='black');im=Image.open(E/f"{r['id']:03}_detail.jpg");im.thumbnail((350,285));sheet.paste(im,(x,y+25));im=Image.open(E/f"{r['id']:03}_crop.png");scale=min(3,350/im.width,130/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)));sheet.paste(im,(x,y+310))
 sheet.save(E/f'sheet_{start:03}.jpg',quality=95)
print('evidence',len(rows))
