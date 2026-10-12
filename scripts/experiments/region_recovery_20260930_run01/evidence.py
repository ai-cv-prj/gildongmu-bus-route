"""Original frame/crop evidence for gains, losses and all newly passed targets."""
from collections import defaultdict
import json
from PIL import Image,ImageDraw
from bus_route.video import sample_frames
from bus_route.target import exact_route,token_quality
from evaluate import ROOT,OUT,BASE,load

def main():
 dest=OUT/'evidence';dest.mkdir(exist_ok=False)
 records=[];grouped=defaultdict(list)
 for mode in ['offline','realtime']:
  ev=json.loads((OUT/f'{mode}_evaluation.json').read_text())
  changes=[{**r,'kind':'gain'} for r in ev['gains']]
  changes += [{**r,'kind':'loss','first_s':r['old_first_s']} for r in ev['losses']]
  changes += [{**r,'kind':'new_alternative','first_s':r['first']['first_s']} for r in ev['new_alternative_flags']]
  changes += [{**r,'kind':'removed_alternative','first_s':r['old_first']['first_s']} for r in ev['removed_alternative_flags']]
  for change in changes:
   old=change['kind'] in {'loss','removed_alternative'}
   data=load((BASE if old else OUT)/mode/change['folder'])
   first=change['first_s']
   ps=[p for p in data[1] if float(p['timestamp_s'])==first and p['eligible'] and exact_route(p['text'],change['target'])]
   if change.get('first') and not old:ps=[p for p in ps if str(change['first']['track_id'])==p['track_id']]
   p=max(ps,key=lambda p:token_quality(p['token_scores']))
   recovered=json.loads(p.get('recovered_from') or '{}')
   same_before=load(BASE/mode/change['folder'])
   originals=[q for q in same_before[1] if float(q['timestamp_s'])==first and q['bus_box']==p['bus_box']]
   r={**change,'mode':mode,'id':len(records),'box':p['text_box'],'bus_box':p['bus_box'],
      'ocr_text':p['text'],'score':token_quality(p['token_scores']),'recovered_from':recovered,
      'previous_same_frame':[{'text':q['text'],'box':q['text_box'],'eligible':q['eligible'],'reason':q['rejection_reason']} for q in originals],
      'verdict':'needs_review','review_note':''}
   records.append(r);grouped[r['video_path']].append(r)
 for path,rows in grouped.items():
  reader=sample_frames(ROOT/path,5);next(reader)
  try:
   for t,rgb in reader:
    for r in rows:
     if abs(t-r['first_s'])>1e-6:continue
     prefix=f'{r["id"]:03}'
     im=Image.fromarray(rgb);im.crop(r['box']).save(dest/f'{prefix}_crop.png')
     source=r['recovered_from'].get('box')
     if source:im.crop(source).save(dest/f'{prefix}_source.png')
     draw=ImageDraw.Draw(im);draw.rectangle(r['bus_box'],outline='orange',width=3);draw.rectangle(r['box'],outline='red',width=3)
     if source:draw.rectangle(source,outline='cyan',width=2)
     a,b,c,d=r['box'];pad=max(c-a,d-b)
     zoom=im.crop((max(0,a-pad),max(0,b-pad),min(im.width,c+pad),min(im.height,d+pad)))
     zoom.thumbnail((850,700));zoom.save(dest/f'{prefix}_detail.jpg',quality=95)
     im.thumbnail((720,960));im.save(dest/f'{prefix}_context.jpg',quality=95)
    if t>=max(r['first_s'] for r in rows):break
  finally:reader.close()
 (dest/'review.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 for start in range(0,len(records),9):
  sheet=Image.new('RGB',(1440,1500),'#eee');draw=ImageDraw.Draw(sheet)
  for j,r in enumerate(records[start:start+9]):
   x=j%3*480;y=j//3*500
   draw.text((x+5,y+5),f'{r["id"]} {r["mode"]} {r["kind"]} {r["folder"]} {r["target"]}',fill='black')
   im=Image.open(dest/f'{r["id"]:03}_detail.jpg');im.thumbnail((465,320));sheet.paste(im,(x+5,y+30))
   im=Image.open(dest/f'{r["id"]:03}_crop.png');factor=min(3,465/im.width,130/im.height);im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))))
   sheet.paste(im,(x+5,y+365))
  sheet.save(dest/f'sheet_{start:03}.jpg',quality=95)
 print('evidence records',len(records),flush=True)
if __name__=='__main__':main()
