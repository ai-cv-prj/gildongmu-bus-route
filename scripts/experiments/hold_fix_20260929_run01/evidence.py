"""Extract only new matches and new alternative flags for visual inspection."""
import csv,json
from pathlib import Path
from PIL import Image,ImageDraw
from bus_route.video import sample_frames
from bus_route.target import exact_route,token_quality
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/hold_fix_20260929_run01';DEST=OUT/'evidence';DEST.mkdir(exist_ok=True)
s=json.loads((OUT/'comparison_summary.json').read_text());items=[]
for r in s['gains']:items.append({**r,'time_s':r['first_match_s'],'kind':'gain'})
for r in json.loads((OUT/'new_alternative_flags.json').read_text()):items.append({**r,'kind':'alternative'})
records=[]
for n,r in enumerate(items):
 with (OUT/r['folder']/'predictions.csv').open(encoding='utf-8-sig') as f:preds=list(csv.DictReader(f))
 candidates=[p for p in preds if float(p['timestamp_s'])==r['time_s'] and exact_route(p['text'],r['target']) and p['eligible']=='True' and token_quality(json.loads(p['token_scores']))>=.9 and ('track_id' not in r or str(r['track_id'])==p['track_id'])]
 p=max(candidates,key=lambda p:token_quality(json.loads(p['token_scores'])));box=json.loads(p['text_box']);reader=sample_frames(ROOT/r['video_path'],5);next(reader)
 try:
  for t,rgb in reader:
   if abs(t-r['time_s'])<1e-6:
    im=Image.fromarray(rgb);crop=f'{n:02d}_crop.png';context=f'{n:02d}_context.jpg';im.crop(box).save(DEST/crop);ImageDraw.Draw(im).rectangle(box,outline='red',width=5);im.thumbnail((640,900));im.save(DEST/context,quality=93);records.append({**r,'box':box,'crop':crop,'context':context});break
 finally:reader.close()
(DEST/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
for start in range(0,len(records),6):
 subset=records[start:start+6];im=Image.new('RGB',(1200,1000),'#eee');draw=ImageDraw.Draw(im)
 for j,r in enumerate(subset):
  x=j%3*400;y=j//3*500;draw.text((x+3,y+3),f'{start+j} {r["kind"]} {r["folder"]} {r["target"]} t={r["time_s"]}',fill='black');ctx=Image.open(DEST/r['context']);ctx.thumbnail((390,370));im.paste(ctx,(x+3,y+25));c=Image.open(DEST/r['crop']);scale=min(380/c.width,90/c.height,4);c=c.resize((max(1,round(c.width*scale)),max(1,round(c.height*scale))));im.paste(c,(x+3,y+405))
 im.save(DEST/f'sheet_{start//6:02d}.jpg',quality=95)
print('evidence',len(records))
