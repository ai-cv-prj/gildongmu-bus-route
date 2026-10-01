"""Bus-wide draft annotations for visual review, never inferred ground truth."""
import json,sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];OLD=ROOT/'output/route_detector_20260930_run01';O=ROOT/'output/bus_roi_finetune_20260930_run01'
(O/'images').mkdir();(O/'sheets').mkdir()
rs={r['key']:r for r in json.loads((OLD/'proposals.json').read_text())};ss=json.loads((OLD/'dataset_reviewed/manifest.json').read_text());out=[];skipped=[]
def intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
for s in ss:
 if s['kind']=='negative':continue
 r=rs[s['key']];b=s['box_original'];area=(b[2]-b[0])*(b[3]-b[1]);buses=[bus['bus_box'] for bus in r['buses'] if intersection(b,bus['bus_box'])/area>=.98]
 if not buses:skipped.append(s['key']);continue
 roi=min(buses,key=lambda b:(b[2]-b[0])*(b[3]-b[1]));im=Image.open(OLD/r['frame_path']).crop(roi)
 boxes=[[a+s['roi'][0]-roi[0],y+s['roi'][1]-roi[1],c+s['roi'][0]-roi[0],z+s['roi'][1]-roi[1]] for a,y,c,z in s['boxes']]
 boxes=[[max(0,a),max(0,y),min(im.width,c),min(im.height,z)] for a,y,c,z in boxes if c>0 and z>0 and a<im.width and y<im.height]
 key=s['key'];im.save(O/'images'/f'{key}.png');out.append(dict(**{k:s[k] for k in ['key','video_path','video_index','split','visible_text','baseline_failed','timestamp_s']},roi=roi,boxes=boxes,primary=boxes[0],width=im.width,height=im.height,image=f'images/{key}.png',review_status='draft'))
(O/'draft.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));(O/'skipped.json').write_text(json.dumps(skipped));print(len(out),'skipped',skipped)
for start in range(0,len(out),12):
 sheet=Image.new('RGB',(1600,520*3),'#ddd');draw=ImageDraw.Draw(sheet)
 for j,s in enumerate(out[start:start+12]):
  im=Image.open(O/s['image']);scale=min(390/im.width,475/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)));di=ImageDraw.Draw(im)
  for b in s['boxes']:di.rectangle([v*scale for v in b],outline='red',width=2)
  x=j%4*400;y=j//4*520;sheet.paste(im,(x,y+38));draw.text((x+3,y+3),f"{s['key']} {s['split']} {s['width']}x{s['height']} scale{scale:.3f}",fill='black')
 sheet.save(O/'sheets'/f'draft_{start:03}.jpg',quality=95)
