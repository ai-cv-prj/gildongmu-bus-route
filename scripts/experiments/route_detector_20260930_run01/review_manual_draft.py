import json,sys
from pathlib import Path
from PIL import Image,ImageDraw
O=Path('output/route_detector_20260930_run01');r=json.loads((O/'proposals.json').read_text())
if (O/'manual_draft.json').exists():
 raise FileExistsError('Preserve the existing draft; choose a new experiment directory.')
manual={2:(0,[354,959,379,978]),14:(1.2,[737,459,808,518]),19:(0,[808,419,846,454]),29:(2,[235,931,304,958]),47:(4.2,[13,567,157,680]),48:(2.6,[78,651,207,755]),55:(5,[675,450,935,515]),59:(8.2,[464,665,514,702]),62:(2.6,[355,300,553,370]),103:(9.8,[934,768,1004,868]),135:(1.2,[417,143,640,239])}
labels=[]
for i in sorted({x['video_index'] for x in r if x['failed']}):
 if i in manual:
  t,b=manual[i];rec=next(x for x in r if x['video_index']==i and x['timestamp_s']==t)
 else:
  rec=next(x for x in r if x['video_index']==i and x['proposal']);b=rec['proposal']
 labels.append(dict(video_index=i,key=rec['key'],bbox=b,visible_text={108:'6716',123:'030',127:'072',128:'072',131:'085',132:'085',39:'03',74:'03',76:'04',97:'13'}.get(i,rec['target']),visibility='uncertain' if i in [55,98,130] else 'readable',note='target location unresolved; visible other number, original target unchanged' if i==108 else ''))
(O/'manual_draft.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
for start in range(0,len(labels),12):
 sheet=Image.new('RGB',(1200,260*4),'#ddd');d=ImageDraw.Draw(sheet)
 for j,l in enumerate(labels[start:start+12]):
  rec=next(x for x in r if x['key']==l['key']);im=Image.open(O/rec['frame_path']).crop(l['bbox']);im=im.resize((im.width*min(4,370/max(im.width,1)),im.height*min(4,370/max(im.width,1)))) if False else im
  scale=min(4,370/im.width,210/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)))
  x=j%3*400;y=j//3*260;sheet.paste(im,(x,y+40));d.text((x+5,y+5),f"{l['key']} {l['visible_text']} {l['bbox']}",fill='black')
 sheet.save(O/f'sheets/manual_draft_{start}.jpg')
