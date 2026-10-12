"""Isolated route detector experiment with fixed OCR and decision rules."""
import os,sys,json,csv,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01'
sys.path.insert(0,str(ROOT/'scripts'))
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')
import torch,numpy as np
from ultralytics import YOLO
from bus_route.detector import BusTextDetector,clip_box
from bus_route.recognizer import PARSeqRecognizer
from bus_route.route_evidence import EVIDENCE_FILTER
from bus_route.region_recovery import RECOVERY_POLICY
from infer_target_video import analyze,DUPLICATE_BUS_POLICY
class RouteDetector(BusTextDetector):
 route_display_only=True
 def __init__(self):
  super().__init__('cuda');self.route=YOLO(str(O/'train/weights/best.pt'))
  self.route.predict(np.zeros((960,960,3),np.uint8),device=0,imgsz=960,rect=False,conf=.25,iou=.5,verbose=False)
  self.craft_fallback_on_empty=False
  self.last_region_source='route_display'
 def text_regions(self,rgb,bus_box):
  x,y,z,w=bus_box;roi=rgb[y:w,x:z]
  result=self.route.predict(roi[:,:,::-1].copy(),device=0,imgsz=960,rect=False,conf=.25,iou=.5,verbose=False)[0]
  boxes=[]
  for b in result.boxes.xyxy.cpu().tolist():
   a,c,d,e=clip_box(b,z-x,w-y)
   if d>a and e>c:boxes.append((a+x,c+y,d+x,e+y))
  if not boxes and self.craft_fallback_on_empty:
   # The route detector remains the first pass. Use the existing general text
   # detector only when it found no region on this particular bus/frame.
   self.last_region_source='craft'
   return super().text_regions(rgb,bus_box)
  self.last_region_source='route_display'
  return bus_box,boxes
 def regions_in_box(self,rgb,search_box,canvas_size=None):
  if self.craft_fallback_on_empty and self.last_region_source=='craft':
   return super().regions_in_box(rgb,search_box,canvas_size=canvas_size)
  return [] # Full bus already searched exactly once; no CRAFT second pass.
if __name__=='__main__':
 assert not (O/'offline').exists() and not (O/'realtime').exists()
 shutil.copyfile(ROOT/'output/region_recovery_20260930_run01/inputs.csv',O/'inputs.csv')
 with (O/'inputs.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 assert len(rows)==144
 files=['scripts/infer_target_video.py','core/bus_route/target.py','core/bus_route/route_evidence.py','core/bus_route/region_recovery.py','core/bus_route/detector.py','core/bus_route/recognizer.py','core/bus_route/video.py','scripts/experiments/bus_roi_finetune_20260930_run01/run_videos.py','output/bus_roi_finetune_20260930_run01/train/weights/best.pt','output/bus_roi_finetune_20260930_run01/inputs.csv']
 hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files};(O/'video_provenance.json').write_text(json.dumps(hashes,indent=2))
 torch.set_num_threads(4);detector=RouteDetector();recognizer=PARSeqRecognizer()
 for mode in ['offline','realtime']:
  out=O/mode;out.mkdir();summaries=[]
  settings=dict(mode=mode,detector='fine-tuned YOLO11n full bus, conf=.25,iou=.5,imgsz=960',sample_fps=5,decision='unchanged matcher, single_score=.98, max realtime age300ms',evidence_filter=EVIDENCE_FILTER,region_recovery=RECOVERY_POLICY,duplicate_bus_recovery=DUPLICATE_BUS_POLICY,legacy_log_names='craft_ms means route detector time; upper search_zone means full bus. body pass is empty.',recognizer=recognizer.metadata(),dataset_note='143 existing development clips incl training sources plus7017->7016 counter; not independent generalization')
  (out/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
  for i,row in enumerate(rows,1):
   print('START',mode,i,144,row['video_path'],flush=True)
   summary=analyze(ROOT/row['video_path'],out/f'video_{i:02}',detector,recognizer,5,row['target_route'],realtime=mode=='realtime',single_score=.98)
   summaries.append(summary);(out/'progress.json').write_text(json.dumps(dict(completed=i,recognized=sum(s['first_recognized_s'] is not None for s in summaries[:143]),last=summary['target_route']),indent=2))
  (out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2));print('COMPLETE',mode,flush=True)
  assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
