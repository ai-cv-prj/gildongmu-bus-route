"""Validate experiment artifacts, preserved code, split integrity and local report links."""
import csv,hashlib,json
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01';W=ROOT/'workbench/bus_roi_finetune_20260930_run01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
for name,path in [('production_baseline','output/region_recovery_20260930_run01/provenance.json'),('video_run','output/bus_roi_finetune_20260930_run01/video_provenance.json')]:
 hashes=json.loads((ROOT/path).read_text());assert all(sha(ROOT/p)==v for p,v in hashes.items());checks[name+'_hashes_unchanged']=len(hashes)
assert sha(ROOT/'output/route_detector_20260930_run01/train_run/weights/best.pt')=='967b2f0f40b8543100cc40d50a54a7fa85b4a8ff284f27afb1d58c691d12303c';checks['initial_weights_preserved']=True
protocol=json.loads((O/'protocol.json').read_text());manifest=O/'dataset_reviewed/manifest.json';assert sha(manifest)==protocol['manifest_sha256'];records=json.loads(manifest.read_text());splits=defaultdict(set);hashes=defaultdict(set)
for r in records:
 im=O/'dataset_reviewed'/r['image'];assert sha(im)==r['sha256'];hashes[r['sha256']].add(r['split'])
 if r.get('video_path'):splits[r['video_path']].add(r['split'])
 for x1,y1,x2,y2 in r['boxes']:assert 0<=x1<x2<=r['width'] and 0<=y1<y2<=r['height']
assert all(len(s)==1 for s in splits.values());assert all(len(s)==1 for s in hashes.values());checks['split_counts']=dict(Counter(r['split'] for r in records));checks['split_source_and_hash_disjoint']=True
inputs=list(csv.DictReader((O/'inputs.csv').open()));assert len(inputs)==144
for mode in ['offline','realtime']:
 e=json.loads((O/f'{mode}_comparison.json').read_text());summary=json.loads((O/mode/'summary.json').read_text());assert len(e['rows'])==len(summary)==144
 assert sum(r['new_recognized'] for r in e['rows'][:143])==e['new_recognized'];assert e['old_recognized']==110
 assert [r['video_path'] for r in e['rows']]==[r['video'] for r in summary]
 assert not e['negative_control']['new_recognized'];assert all(not p['recognized'] for p in e['plate_phone_probes']);checks[mode]=dict(videos=144,active_recognized=e['new_recognized'],negative_controls_passed=True)
rows=json.loads((O/'video_evidence/review.json').read_text());assert all(r['verdict'] in ['route_display','lost_route_display','false_recognition','other_bus_route','uncertain'] and r.get('review_note') for r in rows)
for r in rows:
 for suffix in ['crop.png','context.jpg','detail.jpg']:assert (O/f"video_evidence/{r['id']:03}_{suffix}").is_file()
checks['reviewed_evidence_cases']=len(rows);testlog=(O/'tests.log').read_text();assert 'Ran 44 tests' in testlog and testlog.rstrip().endswith('OK');checks['tests_passed']=44
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  self.links += [v for k,v in attrs if k in ['href','src'] and v]
count=0
for report in [W/'review.html',ROOT/'workbench/index.html']:
 parser=Links();parser.feed(report.read_text())
 for link in parser.links:
  u=urlsplit(link)
  if u.scheme or u.netloc or not u.path:continue
  p=(report.parent/unquote(u.path)).resolve()
  if p==O/'artifact_validation.json':continue
  assert p.exists(),str(p);count+=1
checks['local_links_checked']=count;checks['trained_weights_sha256']=sha(O/'train/weights/best.pt')
(O/'artifact_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));print(json.dumps(checks,ensure_ascii=False,indent=2))
