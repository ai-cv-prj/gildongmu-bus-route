"""Compare experimental full-video runs against unchanged 110/143 production baseline."""
import json,sys,re,csv
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01';BASE=ROOT/'output/region_recovery_20260930_run01'
sys.path.insert(0,str(ROOT/'scripts/experiments/plate_filter_20260929_run01'));from evaluate import load,replay,stats
from bus_route.target import exact_route,token_quality

def compare(mode):
 now=json.loads((O/mode/'summary.json').read_text());old=json.loads((BASE/mode/'summary.json').read_text());known={s['target_route'] for s in old[:143]};rows=[];newflags=[];removed=[];timings={k:[] for k in ['pipeline_ms','frame_age_ms','bus_ms','craft_ms','ocr_ms']};probes=[]
 for i,(n,b) in enumerate(zip(now,old),1):
  assert (n['video'],n['target_route'])==(b['video'],b['target_route']);folder=f'video_{i:02}';data=load(O/mode/folder);before=load(BASE/mode/folder)
  assert bool(replay(data,n['target_route']))==(n['first_recognized_s'] is not None)
  row=dict(folder=folder,video_path=n['video'],target=n['target_route'],old_recognized=b['first_recognized_s'] is not None,new_recognized=n['first_recognized_s'] is not None,old_first_s=b['first_recognized_s'],first_s=n['first_recognized_s'],dropped_samples=n['dropped_samples'],sample_count=n['sample_count']);rows.append(row)
  if i==144:continue
  for f in data[0]:
   for k in timings:
    if f.get(k):timings[k].append(float(f[k]))
  candidates=set(known)
  key=re.sub('^[가-힣]+','',n['target_route'])
  for j,c in enumerate(key):
   if c.isdigit():
    for delta in [-1,1]:candidates.add(key[:j]+str((int(c)+delta)%10)+key[j+1:])
  for p in before[1]+data[1]:
   if re.fullmatch(r'(?:M|N)?[0-9]{2,4}[AB]?',p['text']):candidates.add(p['text'])
  eligible={p['text'] for p in before[1]+data[1] if p['eligible'] and token_quality(p['token_scores'])>=.9}
  for target in sorted(candidates):
   if exact_route(target,n['target_route']) or not any(exact_route(t,target) for t in eligible):continue
   current=replay(data,target);previous=replay(before,target)
   case=dict(folder=folder,video_path=n['video'],target=target,first=current,old_first=previous)
   if current and not previous:newflags.append(case)
   if previous and not current:removed.append(case)
  for target in (['7284'] if i==13 else ['1577','9350'] if i==138 else []):
   raw=[p for p in data[1] if exact_route(p['text'],target)];probes.append(dict(folder=folder,target=target,raw=len(raw),eligible=sum(p['eligible'] for p in raw),recognized=bool(replay(data,target))))
  if i%20==0:print(mode,i,flush=True)
 result=dict(mode=mode,old_recognized=sum(r['old_recognized'] for r in rows[:143]),new_recognized=sum(r['new_recognized'] for r in rows[:143]),gains=[r for r in rows[:143] if r['new_recognized'] and not r['old_recognized']],losses=[r for r in rows[:143] if r['old_recognized'] and not r['new_recognized']],negative_control=rows[-1],processed_frames=sum(s['sample_count'] for s in now[:143]),dropped_samples=sum(s['dropped_samples'] for s in now[:143]),timing={k:stats(v) for k,v in timings.items()},age_over300=sum(v>300 for v in timings['frame_age_ms']),new_alternative_flags=newflags,removed_alternative_flags=removed,plate_phone_probes=probes,rows=rows)
 (O/f'{mode}_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(mode,'recognized',result['new_recognized'],'gains',len(result['gains']),'losses',len(result['losses']),'new_alternatives',len(newflags),flush=True)
if __name__=='__main__':compare(sys.argv[1])
