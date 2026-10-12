"""Fresh raw-source offline, then paced replay, on a single GPU."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/region_recovery_20260930_run01'
files=['core/bus_route/region_recovery.py','scripts/infer_target_video.py','core/bus_route/route_evidence.py','core/bus_route/target.py','core/bus_route/detector.py','core/bus_route/recognizer.py','core/bus_route/video.py','output/region_recovery_20260930_run01/inputs.csv','scripts/tests/test_region_recovery.py']
provenance={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
with (OUT/'provenance.json').open('x') as f:json.dump(provenance,f,indent=2)
for mode in ['offline','realtime']:
 cmd=[sys.executable,str(ROOT/'scripts/infer_target_video.py'),'--videos-csv',str(OUT/'inputs.csv'),'--target-from-manifest','--sample-fps','5','--single-score','.98','--output',str(OUT/mode),'--device','cuda']
 if mode=='realtime':cmd+=['--realtime']
 print('start',mode,flush=True)
 with (OUT/f'{mode}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
 assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in provenance.items())
 with (OUT/f'{mode}_evaluation.log').open('x') as f:
  subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),mode],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
 print('complete',mode,flush=True)
