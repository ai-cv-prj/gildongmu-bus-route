import csv,json,re
from pathlib import Path
from difflib import SequenceMatcher
from PIL import Image,ImageDraw
from bus_route.video import sample_frames
from bus_route.detector import ROOT
out=ROOT/'output/diagnosis_20260929_run01';out.mkdir(exist_ok=True)
r=ROOT/'output/target_20260929_run02'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
comp={x['video_path']:x for x in read(r/'comparison.csv')};records=[]
for row in read(ROOT/'workbench/dataset2_stop_review.csv'):
 if row['stop_status']!='stopped':continue
 info=comp[row['video_path']];folder=info['folder'];preds=read(r/folder/'predictions.csv');frames=read(r/folder/'frames.csv')
 key=re.sub('^[가-힣]+','',row['target_route'])
 candidates=sorted(preds,key=lambda p:(SequenceMatcher(None,p['text'],key).ratio(),min(json.loads(p['token_scores']))),reverse=True)
 times=[]
 for p in candidates:
  t=float(p['timestamp_s'])
  if t not in times:times.append(t)
  if len(times)==2:break
 last=float(frames[-1]['timestamp_s'])
 for t in [last/2//.2*.2,last]:
  t=round(t,1)
  if t not in times:times.append(t)
 times=sorted(times[:2]+[last]) if len(times)>=2 else sorted(set(times))
 times=sorted(set(times));chosen={t:next((p for p in candidates if float(p['timestamp_s'])==t),None) for t in times}
 reader=sample_frames(ROOT/row['video_path'],5);next(reader);cells=[]
 for t,rgb in reader:
  if t not in chosen:continue
  prefix=f'{folder}_{t:.1f}';im=Image.fromarray(rgb);im.save(out/f'{prefix}_frame.png');context=im.copy();p=chosen[t]
  crop=None
  if p:
   box=json.loads(p['text_box']);ImageDraw.Draw(context).rectangle(box,outline='red',width=5)
   crop=im.crop(box);crop.save(out/f'{prefix}_crop.png')
  context.thumbnail((460,740));context.save(out/f'{prefix}_context.jpg',quality=92)
  cells.append((t,p,context,crop))
 reader.close()
 sheet=Image.new('RGB',(480*len(cells),960),'#eee');draw=ImageDraw.Draw(sheet)
 for i,(t,p,context,crop) in enumerate(cells):
  x=i*480;draw.text((x+5,5),f'{folder} t={t:.1f} target={key}',fill='black');draw.text((x+5,25),f'OCR={p["text"] if p else "none"} eligible={p["eligible"] if p else ""}',fill='black')
  sheet.paste(context,(x+5,50))
  if crop:
   scale=min(450/crop.width,120/crop.height,6);crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))));sheet.paste(crop,(x+5,810))
 sheet.save(out/f'{folder}_sheet.jpg',quality=94)
 records.append({'folder':folder,'video_path':row['video_path'],'target_route':row['target_route'],'user_note':row['note'],'times':times})
(out/'inputs.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print('extracted',len(records))
