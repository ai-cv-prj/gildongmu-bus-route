"""Compare fresh inference against plate_filter, including the union of alternative targets."""
import argparse,csv,json,re,importlib.util,hashlib
from collections import Counter,defaultdict
from pathlib import Path
from bus_route.target import exact_route,token_quality
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/region_recovery_20260930_run01'
BASE=ROOT/'output/plate_filter_20260929_run01'
spec=importlib.util.spec_from_file_location('previous_evaluate',ROOT/'scripts/experiments/plate_filter_20260929_run01/evaluate.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
load,replay,stats,write=helper.load,helper.replay,helper.stats,helper.write

def evaluate(mode):
 summaries=json.loads((OUT/mode/'summary.json').read_text())
 olds=json.loads((BASE/mode/'summary.json').read_text())
 assert len(summaries)==len(olds)==144
 known={s['target_route'] for s in summaries[:143]}
 reviews=json.loads((BASE/'evidence/review.json').read_text())
 rows=[];flags=[];new_flags=[];removed_flags=[];diagnostics=[];anchors=[];recoveries=[]
 times=defaultdict(list);counts=Counter();old_observations_lost=[]
 for i,(s,old) in enumerate(zip(summaries,olds),1):
  folder=f'video_{i:02}'
  assert (s['video'],s['target_route'])==(old['video'],old['target_route'])
  data=load(OUT/mode/folder);before=load(BASE/mode/folder)
  recognized=s['first_recognized_s'] is not None;previous=old['first_recognized_s'] is not None
  assert bool(replay(data,s['target_route']))==recognized
  row=dict(folder=folder,video_path=s['video'],target=s['target_route'],old_recognized=previous,new_recognized=recognized,old_first_s=old['first_recognized_s'],first_s=s['first_recognized_s'],sample_count=s['sample_count'],dropped_samples=s['dropped_samples'])
  rows.append(row)
  if i==144:continue
  counts['dropped_samples']+=s['dropped_samples']
  for f in data[0]:
   for k in ['pipeline_ms','frame_age_ms','bus_ms','craft_ms','ocr_ms']:
    if f.get(k):times[k].append(float(f[k]))
   counts['age_over_300ms']+=bool(f.get('frame_age_ms') and float(f['frame_age_ms'])>300)
  now={(p['timestamp_s'],tuple(p['text_box']),p['text']):p for p in data[1]}
  for p in before[1]:
   key=(p['timestamp_s'],tuple(p['text_box']),p['text'])
   if p['eligible'] and exact_route(p['text'],s['target_route']) and (key not in now or not now[key]['eligible']):
    old_observations_lost.append(dict(folder=folder,timestamp_s=p['timestamp_s'],text=p['text'],box=p['text_box'],reason=now[key]['rejection_reason'] if key in now else 'not_observed'))
  for p in data[1]:
   checks=json.loads(p.get('recovery_checks') or '[]')
   counts['expansion_ocr_crops']+=len(checks)
   counts['confirmation_ocr_crops']+=sum('confirmation' in c for c in checks)
   if p.get('recovered_from'):
    recoveries.append(dict(folder=folder,video_path=s['video'],timestamp_s=p['timestamp_s'],text=p['text'],box=p['text_box'],bus_box=p['bus_box'],track_id=p['track_id'],score=token_quality(p['token_scores']),source=json.loads(p['recovered_from'])))
  for rv in reviews:
   if rv['folder']!=folder or rv['id']>=30:continue
   raw=[p for p in data[1] if exact_route(p['text'],rv['target'])]
   if rv['kind']=='alternative':
    first=replay(data,rv['target']);oldfirst=replay(before,rv['target'])
    diagnostics.append(dict(id=rv['id'],folder=folder,target=rv['target'],verdict=rv['verdict'],raw=len(raw),eligible=sum(p['eligible'] for p in raw),old_recognized=bool(oldfirst),recognized=bool(first),first=first,reasons=dict(Counter(p['rejection_reason'] or 'eligible' for p in raw))))
   else:
    anchor=[p for p in raw if abs(float(p['timestamp_s'])-rv['first_s'])<1e-6]
    anchors.append(dict(id=rv['id'],folder=folder,target=rv['target'],anchor_s=rv['first_s'],raw=len(anchor),eligible=sum(p['eligible'] for p in anchor),recognized=bool(replay(data,rv['target']))))
  alternatives=set(known)
  key=re.sub('^[가-힣]+','',s['target_route'])
  for j,c in enumerate(key):
   if c.isdigit():
    for delta in [-1,1]:alternatives.add(key[:j]+str((int(c)+delta)%10)+key[j+1:])
  for p in [*before[1],*data[1]]:
   if re.fullmatch(r'(?:M|N)?[0-9]{2,4}[AB]?',p['text']):alternatives.add(p['text'])
  eligible={p['text'] for p in [*before[1],*data[1]] if p['eligible'] and token_quality(p['token_scores'])>=.9}
  for target in sorted(alternatives):
   if exact_route(target,s['target_route']):continue
   counts['alternative_cases']+=1
   if not any(exact_route(t,target) for t in eligible):continue
   first=replay(data,target);oldfirst=replay(before,target)
   flag=dict(folder=folder,video_path=s['video'],target=target,first=first,old_first=oldfirst)
   if first:flags.append(flag)
   if first and not oldfirst:new_flags.append(flag)
   if oldfirst and not first:removed_flags.append(flag)
  if i%20==0:print(mode,i,flush=True)
 result=dict(mode=mode,old_recognized=sum(r['old_recognized'] for r in rows[:143]),new_recognized=sum(r['new_recognized'] for r in rows[:143]),gains=[r for r in rows[:143] if r['new_recognized'] and not r['old_recognized']],losses=[r for r in rows[:143] if r['old_recognized'] and not r['new_recognized']],negative_control=rows[-1],timing={k:stats(v) for k,v in times.items()},processed_frames=len(times['pipeline_ms']),counts=dict(counts),new_alternative_flags=new_flags,removed_alternative_flags=removed_flags,alternative_passed=len(flags),reviewed_diagnostics=diagnostics,protected_anchors=anchors,lost_target_observations=old_observations_lost,recoveries=recoveries)
 write(OUT/f'{mode}_comparison.csv',rows)
 (OUT/f'{mode}_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 (OUT/f'{mode}_alternative_flags.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2))
 print(json.dumps({k:result[k] for k in ['mode','new_recognized','gains','losses','negative_control','timing','counts','new_alternative_flags','removed_alternative_flags']},ensure_ascii=False,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['offline','realtime']);evaluate(p.parse_args().mode)
