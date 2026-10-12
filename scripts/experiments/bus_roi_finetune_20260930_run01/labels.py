"""Additional route-display boxes selected by visual review in draft sheet coordinates.
Numbers below identify annotation source images, not inference exceptions.
"""
import json,hashlib,shutil
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01';OLD=ROOT/'output/route_detector_20260930_run01';D=O/'dataset';D.mkdir()
ss=json.loads((O/'draft.json').read_text())
# Coordinates within each displayed thumbnail, excluding its 38px caption.
extra={1:[[122,86,251,144]],8:[[137,54,166,91]],11:[[134,143,155,177]],13:[[77,134,104,172]],23:[[96,25,179,75]],30:[[232,163,277,223]],33:[[124,82,164,108]],45:[[27,97,114,141]],49:[[105,337,127,350]],53:[[174,143,214,169]],68:[[0,139,58,175]],105:[[108,270,170,300]],113:[[204,121,244,153]],121:[[151,104,221,147]],125:[[222,94,267,130]],126:[[148,185,208,215]],127:[[181,36,239,80]],128:[[101,58,174,105]],132:[[124,32,155,58]],141:[[161,123,225,166]]}
# Values are reviewed spatial annotations; clipped/ambiguous text is not guessed.
# v53 unlit reflected window, v68 left-clipped LED, v8 scanning/uncertain LED are excluded.
for i in [53,68,8]:extra.pop(i,None)
for s in ss:
 scale=min(390/s['width'],475/s['height'])
 for b in extra.get(s['video_index'],[]):
  box=[round(v/scale) for v in b];box=[max(0,box[0]),max(0,box[1]),min(s['width'],box[2]),min(s['height'],box[3])];s['boxes'].append(box)
 s['review_status']='visually reviewed full bus; complete visible route displays; ambiguous/clipped text not guessed'
 for folder in ['images','labels']:(D/folder/s['split']).mkdir(parents=True,exist_ok=True)
 shutil.copyfile(O/s['image'],D/'images'/s['split']/f"{s['key']}.png")
 s['image']=f"images/{s['split']}/{s['key']}.png";s['kind']='bus'
 # OCR symbols are metadata only; YOLO receives class=0 and coordinates.
 text=''
 for a,b,c,d in s['boxes']:
  assert 0<=a<c<=s['width'] and 0<=b<d<=s['height']
  text+=f"0 {(a+c)/2/s['width']:.8f} {(b+d)/2/s['height']:.8f} {(c-a)/s['width']:.8f} {(d-b)/s['height']:.8f}\n"
 (D/'labels'/s['split']/f"{s['key']}.txt").write_text(text)
# Preserve source-day splits for reviewed non-route number/text backgrounds.
for s in json.loads((OLD/'dataset_reviewed/manifest.json').read_text()):
 if s['kind']!='negative':continue
 s=dict(s);shutil.copyfile(OLD/'dataset_reviewed'/s['image'],D/s['image']);(D/'labels'/s['split']/f"{s['key']}.txt").write_text('');ss.append(s)
(D/'manifest.json').write_text(json.dumps(ss,ensure_ascii=False,indent=2));(D/'data.yaml').write_text(f'path: {D}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: route_display\n')
for split in ['train','val','test']:
 group=[s for s in ss if s['split']==split];print(split,len(group),sum(len(s['boxes']) for s in group))
# Only the new boxes, enlarged for a second coordinate review before training.
new=[]
for s in ss:
 if s['video_index'] in extra and s['kind']=='bus':
  for j,b in enumerate(s['boxes'][-len(extra[s['video_index']]):]):new.append((s,b))
sheet=Image.new('RGB',(1200,200*((len(new)+3)//4)),'#ddd');dr=ImageDraw.Draw(sheet)
for j,(s,b) in enumerate(new):
 im=Image.open(D/s['image']).crop(b);scale=min(4,290/im.width,165/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)));x=j%4*300;y=j//4*200;sheet.paste(im,(x,y+30));dr.text((x,y),s['key']+' '+str(b),fill='black')
sheet.save(O/'sheets/additional_review.jpg')
