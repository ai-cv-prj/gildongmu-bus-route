"""Render a UI-only demo from original frames and already measured realtime decisions.
No new inference, model changes, future-frame interpolation or target-number completion.
"""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'core'))
from bus_route.video import sample_frames
O=ROOT/'workbench/realtime_ui_demo_20260930_run01';B=ROOT/'output/region_recovery_20260930_run01/realtime/video_109'
s=json.loads((B/'summary.json').read_text());source=ROOT/s['video'];target=s['target_route'];rows=list(csv.DictReader((B/'frames.csv').open(encoding='utf-8-sig')))
for r in rows:
 r['time']=float(r['timestamp_s']);r['available']=r['time']+float(r['frame_age_ms'])/1000;r['buses']=json.loads(r['buses'])
def font(size):
 f=ImageFont.truetype(str(O/'font.ttf'),size);f.set_variation_by_axes([700 if size>=27 else 500]);return f
fonts={n:font(n) for n in [19,22,24,27,30,34,40,58]};W,H=720,1280;fps=30;scale=W/s['metadata']['size'][0]
output=O/'sample.mp4';assert not output.exists()
writer=imageio_ffmpeg.write_frames(str(output),(W,H),fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,output_params=['-crf','20','-preset','fast','-movflags','+faststart','-threads','2']);next(writer)
reader=sample_frames(source,fps);next(reader);index=0;latest=None;last_ok=None;last_ok_box=None;track=None;timeline=[];shots={.8:'search',3.0:'recognized',5.7:'checking'}
for t,rgb in reader:
 while index<len(rows) and rows[index]['available']<=t:
  latest=rows[index];index+=1
  good=[b for b in latest['buses'] if b.get('target_decision',{}).get('state') in ['recognized_single','matched_candidate']]
  if good:
   b=good[0];last_ok=latest['time'];track=b['track_id'];last_ok_box=b['bus_box']
 age=float('inf') if last_ok is None else t-last_ok
 current=latest is not None and t-latest['time']<=.6
 tracked=next((b for b in latest['buses'] if b['track_id']==track),None) if current else None
 state='recognized' if age<=.45 and tracked else 'checking' if age<=1.2 and tracked else 'search'
 im=Image.fromarray(rgb).resize((W,H),Image.Resampling.LANCZOS).convert('RGBA');overlay=Image.new('RGBA',im.size);d=ImageDraw.Draw(overlay)
 teal=(49,236,186,255);white=(245,250,255,255);muted=(177,190,207,255);amber=(255,206,119,255);color=teal if state=='recognized' else amber
 # Label uses the current available tracker box, never a future ground-truth box.
 if state!='search':
  a,b,c,e=[int(v*scale) for v in tracked['bus_box']];a=max(6,a);c=min(W-6,c);b=max(6,b);e=min(H-6,e)
  d.rounded_rectangle((a,b,c,e),radius=14,outline=color,width=4)
  label=f'{target}  '+('번호 인식' if state=='recognized' else '최근 인식')
  box=d.textbbox((0,0),label,font=fonts[27]);lw=box[2]-box[0]+34;lh=53;x=max(14,min(W-lw-14,a));y=max(220,b-lh-10)
  d.line((x+min(lw/2,max(16,(a+c)/2-x)),y+lh,(a+c)/2,b),fill=color,width=3)
  d.rounded_rectangle((x,y,x+lw,y+lh),radius=13,fill=color);d.text((x+17,y+10),label,font=fonts[27],fill=(9,27,35,255))
 # Persistent destination + status; no confidence-as-probability or boarding assertion.
 d.rounded_rectangle((22,24,W-22,205),radius=25,fill=(11,22,37,237),outline=(255,255,255,35),width=1)
 d.text((46,42),'찾는 버스',font=fonts[22],fill=muted);d.text((43,76),target,font=fonts[58],fill=white)
 d.rounded_rectangle((W-184,44,W-46,80),radius=18,fill=(36,54,69,255));d.text((W-167,49),'실시간 화면',font=fonts[19],fill=white)
 status={'search':'버스 번호 확인 중','recognized':f'{target}번 버스 인식','checking':'최근 인식 · 다시 확인 중'}[state]
 d.ellipse((48,160,60,172),fill=teal if state=='recognized' else amber if state=='checking' else muted)
 d.text((76,148),status,font=fonts[27],fill=teal if state=='recognized' else amber if state=='checking' else white)
 d.rounded_rectangle((80,H-101,W-80,H-39),radius=23,fill=(11,22,37,217));d.text((W/2,H-88),'버스 방향으로 유지해 주세요',font=fonts[24],fill=white,anchor='mt')
 im=Image.alpha_composite(im,overlay).convert('RGB');writer.send(np.asarray(im));timeline.append(dict(t=round(t,3),state=state,source_result_time=latest['time'] if latest else None,last_recognized_source_time=last_ok,track_id=track if state!='search' else None))
 for st,name in shots.items():
  if abs(t-st)<.001:im.save(O/f'{name}.jpg',quality=94)
writer.close();reader.close()
(O/'timeline.json').write_text(json.dumps(timeline,ensure_ascii=False,indent=2))
provenance=dict(source_video=str(source.relative_to(ROOT)),source_log=str(B.relative_to(ROOT)),target=target,mode='visualization of existing realtime logs; no new inference or network simulation',display=dict(video_fps=fps,source_inference_fps=5,availability='timestamp_s + measured frame_age_ms',current_label_seconds=.45,rechecking_label_seconds=1.2,box='latest available same-track box; no future interpolation',audio=False),hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,B/'frames.csv',B/'decisions.csv',Path(__file__)]},frame_count=len(timeline))
(O/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2));print(json.dumps(dict(output=str(output),frames=len(timeline),states=sorted({r['state'] for r in timeline})),ensure_ascii=False))
