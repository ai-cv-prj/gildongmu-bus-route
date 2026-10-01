"""Extract actual accepted expansion crops from fresh inference for visual review."""
import csv,json
from pathlib import Path
from PIL import Image,ImageDraw
from bus_route.video import sample_frames
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/region_recovery_20260930_run01'
p=OUT/'offline';dest=OUT/'recovery_review';dest.mkdir(exist_ok=True)
records=[]
for folder in sorted(p.glob('video_*')):
 if not (folder/'summary.json').exists():continue
 s=json.loads((folder/'summary.json').read_text())
 rows=[r for r in csv.DictReader((folder/'predictions.csv').open(encoding='utf-8-sig')) if r.get('recovered_from')]
 missing=[]
 for r in rows:
  name=f'{folder.name}_{r["sample_index"]}_{r["candidate_index"]}_{r["text"]}'
  records.append(dict(folder=folder.name,timestamp_s=r['timestamp_s'],text=r['text'],name=name))
  if not (dest/f'{name}_crop.png').exists():missing.append((name,r))
 if not missing:continue
 reader=sample_frames(ROOT/s['video'],5);next(reader)
 try:
  for t,rgb in reader:
   for name,r in missing:
    if abs(t-float(r['timestamp_s']))>1e-6:continue
    im=Image.fromarray(rgb);box=json.loads(r['text_box']);source=json.loads(r['recovered_from'])
    im.crop(box).save(dest/f'{name}_crop.png');im.crop(source['box']).save(dest/f'{name}_source.png')
    draw=ImageDraw.Draw(im);draw.rectangle(box,outline='red',width=3)
    draw.rectangle(json.loads(r['bus_box']),outline='orange',width=3)
    im.thumbnail((600,700));im.save(dest/f'{name}_context.jpg')
   if t>=max(float(r['timestamp_s']) for _,r in missing):break
 finally:reader.close()
for start in range(0,len(records),12):
 sheet=Image.new('RGB',(1200,1200),'#eee');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(records[start:start+12]):
  x=j%3*400;y=j//3*300
  draw.text((x+5,y+5),r['name'],fill='black')
  im=Image.open(dest/(r['name']+'_crop.png'));im.thumbnail((390,255));im=im.resize((min(390,im.width*3),min(255,im.height*3)));sheet.paste(im,(x+5,y+30))
 sheet.save(dest/f'sheet_{start:03}.jpg')
(dest/'records.json').write_text(json.dumps(records,indent=2))
print('accepted',len(records))
