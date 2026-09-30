"""다른 목표에서 통과한 출력의 위치를 복원한다."""
import csv,json
from pathlib import Path
from PIL import Image,ImageDraw
from bus_route.detector import ROOT
from bus_route.target import exact_route
from bus_route.video import sample_frames
r=ROOT/'output/target_20260929_run02';out=r/'alternative_audit';out.mkdir(exist_ok=True)
flags=json.loads((r/'comparison_summary.json').read_text())['counterfactual_flags'];items=[]
for i,f in enumerate(flags):
 p=list(csv.DictReader((r/f['folder']/'predictions.csv').open(encoding='utf-8-sig')))
 candidates=[x for x in p if exact_route(x['text'],f['target_route']) and x['eligible']=='True' and float(x['timestamp_s'])==f['first_s'] and int(x['track_id'])==f['track_id']]
 x=max(candidates,key=lambda x:min(json.loads(x['token_scores'])));box=json.loads(x['text_box']);reader=sample_frames(ROOT/f['video_path'],5);next(reader)
 try:
  for t,rgb in reader:
   if t==f['first_s']:
    im=Image.fromarray(rgb);im.crop(box).save(out/f'{i:02d}_crop.png');ImageDraw.Draw(im).rectangle(box,outline='red',width=5)
    im.thumbnail((450,800));im.save(out/f'{i:02d}_context.jpg',quality=90);break
 finally:reader.close()
 items.append({**f,'crop':f'{i:02d}_crop.png','context':f'{i:02d}_context.jpg','box':box})
for start in range(0,len(items),8):
 im=Image.new('RGB',(1440,1440),'#eee');draw=ImageDraw.Draw(im)
 for j,x in enumerate(items[start:start+8]):
  a=j%4*360;b=j//4*720;draw.text((a+5,b+5),f'{x["folder"]}: target {x["target_route"]} @ {x["first_s"]}s',fill='black')
  ctx=Image.open(out/x['context']);ctx.thumbnail((350,540));im.paste(ctx,(a+5,b+30))
  c=Image.open(out/x['crop']);scale=min(340/c.width,120/c.height,6);c=c.resize((max(1,int(c.width*scale)),max(1,int(c.height*scale))))
  im.paste(c,(a+5,b+585))
 im.save(out/f'sheet_{start//8+1:02d}.jpg',quality=94)
(out/'evidence.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
print('audited',len(items))
