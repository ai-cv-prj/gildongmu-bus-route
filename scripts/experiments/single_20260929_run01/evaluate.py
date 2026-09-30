"""Replay unchanged OCR output through the new causal single-observation policy."""
import csv,json,collections,re,time
from pathlib import Path
from bus_route.target import TargetMatcher,exact_route,token_quality
R=Path(__file__).resolve().parents[3];BASE=R/'output/hold_fix_20260929_run01';OUT=R/'output/single_20260929_run01'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text('');return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(folder):
 fs=read(folder/'frames.csv');ps=read(folder/'predictions.csv');by=collections.defaultdict(list)
 for f in fs:f['buses']=json.loads(f['buses'])
 for p in ps:p['eligible']=p['eligible']=='True';p['token_scores']=json.loads(p['token_scores']);by[p['sample_index']].append(p)
 return fs,ps,by

def replay(data,target,score=.98,all_rows=False):
 fs,ps,by=data;m=TargetMatcher(target,single_score=score);decisions=[];first=None;repeat=None
 for f in fs:
  for i,b in enumerate(f['buses']):
   d=m.update(b['track_id'],float(f['timestamp_s']),[p for p in by[f['sample_index']] if int(p['bus_index'])==i],b['bus_box'],b.get('decision_complete',not b['candidate_truncated']))
   row={'sample_index':f['sample_index'],'timestamp_s':float(f['timestamp_s']),'track_id':b['track_id'],**d}
   if first is None and d['state'] in {'matched_candidate','recognized_single'}:first=row
   if repeat is None and d['state']=='matched_candidate':repeat=row
   if all_rows:decisions.append(row)
 return first,repeat,decisions

if __name__=='__main__':
 summary=json.loads((BASE/'summary.json').read_text());assert len(summary)==144
 rows=[];flags=[];cases=0;timing=[];known={s['target_route'] for s in summary[:143]}
 for i,s in enumerate(summary,1):
  folder=f'video_{i:02d}';data=load(BASE/folder);start=time.perf_counter();first,repeat,decisions=replay(data,s['target_route'],all_rows=True);timing.append((time.perf_counter()-start)*1000)
  assert (repeat is not None)==(s['first_matched_candidate_s'] is not None),(folder,'legacy regression')
  write(OUT/'offline'/folder/'decisions.csv',decisions)
  rows.append(dict(video_path=s['video'],target=s['target_route'],folder=folder,old_recognized=s['first_matched_candidate_s'] is not None,new_recognized=first is not None,first_s=first['timestamp_s'] if first else None,first_state=first['state'] if first else None,track_id=first['track_id'] if first else None,first_repeat_s=repeat['timestamp_s'] if repeat else None,raw_exact=s['first_exact_s'] is not None))
  if i==144:continue
  key=re.sub('^[가-힣]+','',s['target_route']);alternatives=set(known)
  for j,c in enumerate(key):
   if c.isdigit():
    for delta in [-1,1]:alternatives.add(key[:j]+str((int(c)+delta)%10)+key[j+1:])
  for p in data[1]:
   if re.fullmatch(r'(?:M|N)?[0-9]{2,4}[AB]?',p['text']):alternatives.add(p['text'])
  texts={p['text'] for p in data[1] if p['eligible'] and token_quality(p['token_scores'])>=.9}
  for target in sorted(alternatives):
   if exact_route(target,s['target_route']):continue
   cases+=1
   if not any(exact_route(text,target) for text in texts):continue
   first,repeat,_=replay(data,target)
   if first:flags.append(dict(video_path=s['video'],folder=folder,target=target,baseline_also_passed=repeat is not None,first_s=first['timestamp_s'],state=first['state'],track_id=first['track_id']))
  if i%20==0:print('evaluated',i,flush=True)
 write(OUT/'offline/comparison.csv',rows)
 totals=dict(videos=143,old_recognized=sum(r['old_recognized'] for r in rows[:143]),new_recognized=sum(r['new_recognized'] for r in rows[:143]),gains=[r for r in rows[:143] if r['new_recognized'] and not r['old_recognized']],losses=[r for r in rows[:143] if r['old_recognized'] and not r['new_recognized']],negative_control=rows[-1],alternative_cases=cases,alternative_flags=len(flags),new_alternative_flags=sum(not f['baseline_also_passed'] for f in flags),policy_replay_ms_total=sum(timing),note='Cached OCR; video-level recognition with single >=.98 or legacy repetition, not verified precision')
 (OUT/'offline/summary.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2));(OUT/'offline/alternative_flags.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2));print(json.dumps(totals,ensure_ascii=False,indent=2))
