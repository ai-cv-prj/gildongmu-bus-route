"""Manifest, timing/decision integrity, source snapshots, and local report links."""
import csv,json,hashlib
from pathlib import Path
from urllib.parse import unquote,urlsplit
from bs4 import BeautifulSoup
from evaluate import ROOT,OUT,BASE,load

def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=read(OUT/'inputs.csv')
active=read(ROOT/'data/dataset1_active.csv')+read(ROOT/'data/dataset2_active.csv')
assert len(active)==143 and len(inputs)==144
assert [(r['video_path'],r['target_route']) for r in active]==[(r['video_path'],r['target_route']) for r in inputs[:143]]
assert inputs[-1]['target_route']=='7016'
assert next(r for r in active if r['video_path']==inputs[-1]['video_path'])['target_route']=='7017'
old_manifest=json.loads((BASE/'manifest_validation.json').read_text())
assert all(digest(ROOT/p)==sha for p,sha in old_manifest['source_hashes'].items())
assert digest(OUT/'inputs.csv')==digest(BASE/'inputs.csv')
provenance=json.loads((OUT/'provenance.json').read_text())
assert all(digest(ROOT/p)==sha for p,sha in provenance.items())
violations=[];counts={};old_missing=[]
for mode in ['offline','realtime']:
 summaries=json.loads((OUT/mode/'summary.json').read_text())
 assert len(summaries)==144
 counts[mode]=dict(videos=144,samples=0,recognized=0)
 for i,s in enumerate(summaries,1):
  assert (s['video'],s['target_route'])==(inputs[i-1]['video_path'],inputs[i-1]['target_route'])
  data=load(OUT/mode/f'video_{i:02}')
  assert len(data[0])==s['sample_count']
  assert len(data[1])==s['prediction_count']
  samples=[int(f['sample_index']) for f in data[0]]
  assert samples==sorted(set(samples))
  assert s['rendered_frame_count']-len(samples)==s['dropped_samples']
  if i<=143:
   counts[mode]['samples']+=len(samples)
   counts[mode]['recognized']+=s['first_recognized_s'] is not None
  for f in data[0]:
   for b in f['buses']:
    if b['target_decision']['state'] in {'recognized_single','matched_candidate'} and not b['decision_complete']:
     violations.append(dict(mode=mode,folder=i,timestamp=f['timestamp_s']))
  assert (OUT/mode/f'video_{i:02}/annotated.mp4').stat().st_size>0
 assert summaries[-1]['first_recognized_s'] is None
assert not violations
page=ROOT/'workbench/region_recovery_20260930_run01/review.html'
missing=[];checked=0
for tag in BeautifulSoup(page.read_text(),'html.parser').find_all(['a','img']):
 link=tag.get('href') if tag.name=='a' else tag.get('src')
 if not link or urlsplit(link).scheme:continue
 path=unquote(urlsplit(link).path)
 if not path:continue
 checked+=1
 if not (page.parent/path).resolve().exists():missing.append(link)
assert not missing,missing
result=dict(active_videos=143,negative_controls=1,inputs_match_baseline=True,active_csv_hashes_unchanged=True,code_matches_run_hashes=True,counts=counts,recognition_on_incomplete_bus=violations,checked_static_links=checked,missing_links=missing)
(OUT/'artifact_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,indent=2))
