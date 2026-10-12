"""UI revision: recognized route prominent; searching/checking/failure explicit.
Replays existing evidence, independent of the user's requested route; no new inference.
"""
import sys,csv,json,re,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'scripts/experiments/plate_filter_20260929_run01'))
from evaluate import load
from bus_route.target import TargetMatcher
from bus_route.video import sample_frames
O=ROOT/'workbench/realtime_ui_demo_20260930_run02';B=ROOT/'output/region_recovery_20260930_run01/realtime/video_109'
s=json.loads((B/'summary.json').read_text());source=ROOT/s['video'];target=s['target_route'];rows,ps,by=load(B);matchers={}
for r in rows:
 r['time']=float(r['timestamp_s']);r['available']=r['time']+float(r['frame_age_ms'])/1000
 for p in by[r['sample_index']]:
  if p['eligible'] and re.fullmatch(r'(?:M|N)?[0-9]{2,4}[AB]?',p['text']):matchers.setdefault(p['text'],TargetMatcher(p['text'],single_score=.98))
 for i,b in enumerate(r['buses']):
  b['read_numbers']=[text for text,m in matchers.items() if m.update(b['track_id'],r['time'],[p for p in by[r['sample_index']] if int(p['bus_index'])==i],b['bus_box'],complete=b['decision_complete'])['state'] in ['recognized_single','matched_candidate']]
def font(n):
 f=ImageFont.truetype(str(O/'font.ttf'),n);f.set_variation_by_axes([750 if n>=30 else 550]);return f
F={n:font(n) for n in [20,22,24,28,32,36,44,60,108]};W,H=720,1280;scale=W/s['metadata']['size'][0];fps=30
out=O/'sample.mp4';assert not out.exists();writer=imageio_ffmpeg.write_frames(str(out),(W,H),fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,output_params=['-crf','20','-preset','fast','-movflags','+faststart','-threads','2']);next(writer)
reader=sample_frames(source,fps);next(reader);idx=0;latest=None;focus=None;started=None;reads={};timeline=[];shots={2.0:'checking',3.0:'recognized',6.4:'failed'}
for t,rgb in reader:
 while idx<len(rows) and rows[idx]['available']<=t:
  latest=rows[idx];idx+=1
  for b in latest['buses']:
   if len(b['read_numbers'])==1:reads[b['track_id']]=(latest['time'],b['read_numbers'][0])
  if not any(b['track_id']==focus for b in latest['buses']):
   selected=max(latest['buses'],key=lambda b:(b['bus_box'][2]-b['bus_box'][0])*(b['bus_box'][3]-b['bus_box'][1]),default=None)
   focus=selected['track_id'] if selected else None;started=latest['available'] if selected else None
 tracked=next((b for b in latest['buses'] if b['track_id']==focus),None) if latest and t-latest['time']<=.6 else None
 last,number=reads.get(focus,(-999,None));age=t-last
 if not tracked:state='search';number=None
 elif age<=.45:state='recognized'
 elif t-max(started,last)>=1.2:state='failed';number=None
 else:state='checking';number=None
 color={'recognized':(49,236,186,255),'checking':(130,190,255,255),'failed':(255,107,119,255),'search':(187,202,217,255)}[state];white=(245,250,255,255);dark=(11,22,37,244)
 im=Image.fromarray(rgb).resize((W,H),Image.Resampling.LANCZOS).convert('RGBA');layer=Image.new('RGBA',(W,H));d=ImageDraw.Draw(layer)
 if tracked:
  a,b,c,e=[int(v*scale) for v in tracked['bus_box']];a=max(6,a);c=min(W-6,c);b=max(6,b);e=min(H-6,e);d.rounded_rectangle((a,b,c,e),radius=14,outline=color,width=4)
  label=number if state=='recognized' else '버스 번호 확인 중' if state=='checking' else '번호 인식 실패';lf=F[60] if state=='recognized' else F[28];bounds=d.textbbox((0,0),label,font=lf);lw=bounds[2]-bounds[0]+32;lh=86 if state=='recognized' else 54;x=max(14,min(W-lw-14,a));y=max(254,b-lh-9)
  d.line((x+min(lw/2,max(16,(a+c)/2-x)),y+lh,(a+c)/2,b),fill=color,width=3);d.rounded_rectangle((x,y,x+lw,y+lh),radius=12,fill=color);d.text((x+16,y+(lh-(bounds[3]-bounds[1]))/2-bounds[1]),label,font=lf,fill=(9,24,36,255))
 # Recognized number is the primary element; desired route stays secondary.
 d.rounded_rectangle((22,24,W-22,240),radius=24,fill=dark,outline=color,width=2)
 d.text((46,43),'인식한 버스 번호' if state=='recognized' else '버스 번호 인식',font=F[22],fill=white)
 d.text((W-46,43),f'찾는 번호 {target}',font=F[22],fill=(187,202,217,255),anchor='rt')
 if state=='recognized':d.text((42,76),number,font=F[108],fill=color)
 else:
  title={'checking':'버스 번호 확인 중','failed':'번호 인식 실패','search':'버스를 찾고 있어요'}[state];d.text((46,101),title,font=F[44] if state!='checking' else F[36],fill=color)
  sub={'checking':'번호를 읽고 있어요','failed':'다시 확인하고 있어요','search':'버스 방향으로 유지해 주세요'}[state];d.text((48,174),sub,font=F[24],fill=white)
 im=Image.alpha_composite(im,layer).convert('RGB');writer.send(np.asarray(im));timeline.append(dict(t=round(t,3),state=state,number=number,track_id=focus if tracked else None,source_result_time=latest['time'] if latest else None,available=latest['available'] if latest else None))
 for st,name in shots.items():
  if abs(t-st)<.001:im.save(O/f'{name}.jpg',quality=95)
writer.close();reader.close();(O/'timeline.json').write_text(json.dumps(timeline,ensure_ascii=False,indent=2))
prov=dict(source_video=str(source.relative_to(ROOT)),source_log=str(B.relative_to(ROOT)),mode='UI-only replay, no network simulation or new model inference',display=dict(recognized_font_px=108,requested_font_px=22,success_display_seconds=.45,read_timeout_seconds=1.2,timeout_is_ui_proposal=True,focus='largest available bus when current track disappears; maintain same ID while available',read_number='TargetMatcher replay over observed eligible route strings; independent of requested route',available='timestamp_s + measured frame_age_ms',audio=False),hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,B/'frames.csv',B/'predictions.csv',B/'decisions.csv',Path(__file__)]})
(O/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2));print({'frames':len(timeline),'states':sorted({r['state'] for r in timeline})})
