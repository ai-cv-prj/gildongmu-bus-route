"""Compare fixed pipeline with frozen 143-video baseline; replay alternative targets."""
import csv,json,collections,hashlib,html,re,shutil
from pathlib import Path
import numpy as np
from bus_route.target import TargetMatcher,exact_route,token_quality
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/hold_fix_20260929_run01';WORK=ROOT/'workbench/hold_fix_20260929_run01'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def save(name,rows):
 if not rows:return
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def load(folder):
 frames=read(folder/'frames.csv');preds=read(folder/'predictions.csv');by_frame=collections.defaultdict(list)
 for p in preds:
  p['eligible']=p['eligible']=='True';p['token_scores']=json.loads(p['token_scores']);by_frame[p['sample_index']].append(p)
 for f in frames:f['buses']=json.loads(f['buses'])
 return frames,preds,by_frame

def replay(data,target):
 frames,preds,by_frame=data;m=TargetMatcher(target)
 for f in frames:
  for i,b in enumerate(f['buses']):
   obs=[p for p in by_frame[f['sample_index']] if int(p['bus_index'])==i]
   d=m.update(b['track_id'],float(f['timestamp_s']),obs,b['bus_box'],not b['candidate_truncated'])
   if d['state']=='matched_candidate':return {'time_s':float(f['timestamp_s']),'track_id':b['track_id']}
 return None

baseline=read(ROOT/'output/combined_20260929_run01/videos.csv');old={r['video_path']:r for r in baseline};summary=json.loads((OUT/'summary.json').read_text());assert len(summary)==144
rows=[];hold_reasons=[];flags=[];new_flags=[];cases=0;times=[];known={r['target'] for r in baseline}
for i,s in enumerate(summary[:143],1):
 b=old[s['video']];folder=f'video_{i:02d}';data=load(OUT/folder);previous=load(ROOT/'output'/b['run']/b['folder']);frames,preds,_=data;times.extend(float(f['pipeline_ms']) for f in frames)
 new_match=s['first_matched_candidate_s'] is not None;new_raw=s['first_exact_s'] is not None
 row=dict(video_path=s['video'],target=s['target_route'],folder=folder,old_match=b['matched']=='True',new_match=new_match,old_raw=b['raw_exact']=='True',new_raw=new_raw,first_match_s=s['first_matched_candidate_s'])
 rows.append(row)
 if b['category']=='raw_but_held':
  ps=[p for p in previous[1] if exact_route(p['text'],b['target'])];good=[p for p in ps if p['eligible'] and token_quality(p['token_scores'])>=.9]
  counts=collections.Counter(p['rejection_reason'] for p in ps if not p['eligible'])
  ds=read(ROOT/'output'/b['run']/b['folder']/'decisions.csv');reason=collections.Counter(d['reason'] for d in ds)
  hold_reasons.append(dict(video_path=s['video'],target=b['target'],target_observations=len(ps),eligible_high_score_times=len({p['timestamp_s'] for p in good}),rejected_target_reasons=dict(counts),decision_reasons=dict(reason),new_match=new_match))
 # Same alternative set for both versions; does not imply all alternatives are absent in video.
 alternatives=set(known)
 key=re.sub('^[가-힣]+','',b['target'])
 for j,c in enumerate(key):
  if c.isdigit():
   for delta in [-1,1]:alternatives.add(key[:j]+str((int(c)+delta)%10)+key[j+1:])
 for p in previous[1]+preds:
  if re.fullmatch(r'(?:M|N)?[0-9]{2,4}[AB]?',p['text']):alternatives.add(p['text'])
 for target in sorted(alternatives):
  if exact_route(target,b['target']):continue
  # A string never observed at high score cannot satisfy three exact matches.
  def possible(d):return any(exact_route(p['text'],target) and p['eligible'] and token_quality(p['token_scores'])>=.9 for p in d[1])
  cases+=1
  now=replay(data,target) if possible(data) else None
  before=replay(previous,target) if possible(previous) else None
  if now:
   flag=dict(video_path=s['video'],folder=folder,target=target,baseline_also_passed=before is not None,**now);flags.append(flag)
   if before is None:new_flags.append(flag)
 control=summary[-1]
 if i%20==0:print('audited',i,flush=True)
assert len(hold_reasons)==31
save('comparison.csv',rows);dump('prior_31_hold_reasons.json',hold_reasons);dump('alternative_flags.json',flags);dump('new_alternative_flags.json',new_flags)
totals=dict(videos=143,old_match=sum(r['old_match'] for r in rows),new_match=sum(r['new_match'] for r in rows),new_raw=sum(r['new_raw'] for r in rows),gains=[r for r in rows if r['new_match'] and not r['old_match']],losses=[r for r in rows if r['old_match'] and not r['new_match']],prior31_recovered=sum(r['new_match'] for r in hold_reasons),alternative_cases=cases,alternative_flags=len(flags),new_alternative_flags=len(new_flags),negative_7017_as_7016_passed=control['first_matched_candidate_s'] is not None,pipeline_p50_ms=float(np.percentile(times,50)),pipeline_p95_ms=float(np.percentile(times,95)))
dump('comparison_summary.json',totals)
shutil.copyfile(ROOT/'scripts/experiments/hold_fix_20260929_run01/inputs.csv',OUT/'inputs.csv')
for src,name in [('/tmp/bus_hold_fix_inference.log','inference.log'),('/tmp/bus_hold_fix_tests.log','tests.log')]:shutil.copyfile(src,OUT/name)
dump('source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/infer_target_video.py',*sorted((ROOT/'core/bus_route').glob('*.py')),*sorted(Path(__file__).parent.glob('*.py'))]})
WORK.mkdir(exist_ok=True)
print(json.dumps(totals,ensure_ascii=False,indent=2))
