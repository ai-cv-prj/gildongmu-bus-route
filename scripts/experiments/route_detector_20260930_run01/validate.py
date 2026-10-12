"""Check artifact consistency without rerunning or rewriting existing evaluations."""
import json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from bs4 import BeautifulSoup
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/route_detector_20260930_run01';W=ROOT/'workbench/route_detector_20260930_run01'
summary=json.loads((O/'diagnosis_summary.json').read_text());assert len(summary['rows'])==33
assert Counter(r['diagnosis'] for r in summary['rows'])==summary['summary']['categories']
assert sum(r['matches_target'] for r in summary['rows'])==21
assert sum(r['matches_target'] and r['quality']>=.98 for r in summary['rows'])==14
for r in summary['rows']:Image.open(O/r['path']).verify()
manifest=json.loads((O/'dataset_reviewed/manifest.json').read_text());splits=defaultdict(set);dates=defaultdict(set)
for s in manifest:
 splits[s['split']].add(s['video_path']);dates[s['split']].add(Path(s['video_path']).name[:8]);im=Image.open(O/'dataset_reviewed'/s['image']);assert im.size==(s['width'],s['height']);im.verify()
 assert hashlib.sha256((O/'dataset_reviewed'/s['image']).read_bytes()).hexdigest()==s['sha256']
for a in splits:
 for b in splits:
  if a!=b:assert not splits[a]&splits[b] and not dates[a]&dates[b]
old=json.loads((ROOT/'output/region_recovery_20260930_run01/provenance.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in old.items())
e=json.loads((O/'detector_evaluation.json').read_text());assert len(e['local_results'])==27 and len(e['bus_transfer'])==25
assert e['summary']['yolo']['primary_ocr_exact']==23 and e['summary']['craft']['primary_ocr_exact']==15
assert e['automatic_bus_transfer_summary']['yolo_bus']['primary_ocr_exact']==16
assert hashlib.sha256((O/'train_run/weights/best.pt').read_bytes()).hexdigest()==e['weights_sha256']
soup=BeautifulSoup((W/'review.html').read_text(),'html.parser');linked=[]
for tag in soup.find_all(['a','img']):
 ref=tag.get('href') or tag.get('src')
 if ref and not ref.startswith(('http','#')):
  path=(W/ref.split('#')[0]).resolve();assert path.exists(),path;linked.append(str(path))
  if tag.name=='img':Image.open(path).verify()
assert len(soup.select('article[id]'))==33
assert 'Ran 44 tests' in (O/'tests.log').read_text() and (O/'tests.log').read_text().rstrip().endswith('OK')
files=[*Path(__file__).parent.glob('*.py'),O/'manual_labels.json',O/'manual_label_correction.json',O/'protocol.json',O/'dataset_reviewed/manifest.json',O/'ocr_diagnosis.json',O/'ocr_correction.json',O/'detector_evaluation.json',O/'training_provenance.json',W/'review.html']
result=dict(status='passed',diagnostic_videos=33,local_test_images=27,automatic_bus_test_frames=25,local_link_references=len(linked),production_hashes_unchanged=len(old),tests=44,date_splits={k:sorted(v) for k,v in dates.items()},sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(O/'artifact_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False,indent=2))
