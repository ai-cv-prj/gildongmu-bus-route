"""Frozen heldout localization/OCR evaluation, plus automatic bus-ROI transfer check."""
import os,json,time,hashlib,sys
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
import torch
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/route_detector_20260930_run01';D=O/'dataset_reviewed'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')
from ultralytics import YOLO
from bus_route.detector import BusTextDetector,clip_box
from bus_route.recognizer import PARSeqRecognizer
from bus_route.target import exact_route,token_quality
sys.path.insert(0,str(ROOT/'scripts/experiments/region_recovery_20260930_run01'));from evaluate import load
# Import ambiguity is avoided below for direct execution: sibling evaluate may be found first.
def metrics(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1]);return dict(iou=inter/(aa+bb-inter) if aa+bb>inter else 0,coverage=inter/bb if bb else 0,area_ratio=aa/bb if bb else 0)
def stats(xs):return dict(n=len(xs),p50=float(np.percentile(xs,50)),p95=float(np.percentile(xs,95)),max=float(max(xs))) if xs else None
ss=json.loads((D/'manifest.json').read_text());rs={r['key']:r for r in json.loads((O/'proposals.json').read_text())}
model=YOLO(str(O/'train_run/weights/best.pt'));craft=BusTextDetector();ocr=PARSeqRecognizer()
def detect(im,which):
 arr=np.array(im);torch.cuda.synchronize();start=time.perf_counter()
 if which=='yolo':
  pred=model.predict(arr[:,:,::-1].copy(),device=0,imgsz=640,rect=False,conf=.25,iou=.5,verbose=False)[0];boxes=[dict(box=list(clip_box(b,im.width,im.height)),confidence=float(c)) for b,c in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist())]
 else:boxes=[dict(box=list(b),confidence=None) for b in craft.regions_in_box(arr,(0,0,im.width,im.height))]
 torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
 return boxes,ms
def read(im,boxes):
 preds=ocr.predict_batch([im.crop(b['box']) for b in boxes])
 for b,p in zip(boxes,preds):b['ocr']=asdict(p);b['quality']=token_quality(p.token_scores)
 return boxes
for _ in range(3):
 for method in ['yolo','craft']:detect(Image.open(D/ss[0]['image']),method)
results=[]
(O/'detector_evidence').mkdir()
for s in ss:
 if s['split']!='test':continue
 im=Image.open(D/s['image']).convert('RGB');row=dict(sample=s,methods={})
 for method in ['yolo','craft']:
  boxes,ms=detect(im,method);read(im,boxes)
  for b in boxes:
   b['gt_metrics']=[metrics(b['box'],gt) for gt in s['boxes']]
  row['methods'][method]=dict(boxes=boxes,detector_ms=ms,ocr_ms=sum(b['ocr']['elapsed_ms'] for b in boxes))
 # Review images label ground truth green and detector red; CRAFT blue.
 view=im.copy();draw=ImageDraw.Draw(view)
 for gt in s['boxes']:draw.rectangle(gt,outline='lime',width=2)
 for b in row['methods']['yolo']['boxes']:draw.rectangle(b['box'],outline='red',width=2)
 view.save(O/'detector_evidence'/f"{s['key']}_local.png")
 results.append(row);print('local',s['key'],flush=True)
# Automatic ROIs from the baseline tracker, all buses in the frame, no manual region at inference.
transfers=[]
for s in [s for s in ss if s['split']=='test' and s['kind']=='positive']:
 r=rs[s['key']];im=Image.open(O/r['frame_path']).convert('RGB');boxes=[];elapsed=0
 for bus in r['buses']:
  roi=bus['bus_box'];sub,ms=detect(im.crop(roi),'yolo');elapsed+=ms
  for b in sub:
   a,y,c,z=b['box'];b['box']=[a+roi[0],y+roi[1],c+roi[0],z+roi[1]];b['bus_box']=roi;boxes.append(b)
 read(im,boxes)
 for b in boxes:b['gt_metrics']=metrics(b['box'],s['box_original'])
 baseline=r['ocr']
 transfers.append(dict(key=s['key'],sample=s,boxes=boxes,detector_ms=elapsed,ocr_ms=sum(b['ocr']['elapsed_ms'] for b in boxes),baseline_boxes=baseline))
 view=im.copy();draw=ImageDraw.Draw(view);draw.rectangle(s['box_original'],outline='lime',width=3)
 for b in boxes:draw.rectangle(b['box'],outline='red',width=3)
 view.thumbnail((600,800));view.save(O/'detector_evidence'/f"{s['key']}_bus.jpg",quality=95)
 print('bus',s['key'],flush=True)
# Correction selected from a visual context check, not OCR-based crop search.
corrected=[]
for v in ['exact','pad10']:
 path=f'manual_crops/v064_015_review2_{v}.png';p=ocr.predict(Image.open(O/path));corrected.append(dict(video_index=64,variant=v,path=path,prediction=asdict(p),quality=token_quality(p.token_scores),matches_visible=exact_route(p.text,'271'),matches_target=exact_route(p.text,'271')))
(O/'ocr_correction.json').write_text(json.dumps(corrected,ensure_ascii=False,indent=2))
summary={}
for method in ['yolo','craft']:
 gtcount=0;tp=0;predcount=0;primary_recall=0;complete=0;exact=0;strong=0;negpred=0;bad_primary=[]
 for row in results:
  s=row['sample'];bs=row['methods'][method]['boxes'];gtcount+=len(s['boxes']);predcount+=len(bs)
  edges=sorted([(metrics(b['box'],gt)['iou'],i,j) for i,b in enumerate(bs) for j,gt in enumerate(s['boxes'])],reverse=True);matchedp=set();matchedg=set()
  for score,i,j in edges:
   if score>=.5 and i not in matchedp and j not in matchedg:tp+=1;matchedp.add(i);matchedg.add(j)
  if s['kind']=='negative':negpred+=len(bs);continue
  primary=[b for b in bs if b['gt_metrics'][0]['iou']>=.5]
  primary_recall+=bool(primary);complete+=any(b['gt_metrics'][0]['coverage']>=.95 and b['gt_metrics'][0]['area_ratio']<=2 for b in bs)
  exact+=any(exact_route(b['ocr']['text'],s['visible_text']) for b in primary);strong+=any(exact_route(b['ocr']['text'],s['visible_text']) and b['quality']>=.98 for b in primary)
  for b in primary:
   if not exact_route(b['ocr']['text'],s['visible_text']):bad_primary.append(dict(key=s['key'],expected=s['visible_text'],text=b['ocr']['text'],quality=b['quality'],box=b['box']))
 summary[method]=dict(gt_boxes=gtcount,predictions=predcount,tp_iou50=tp,precision=tp/predcount if predcount else 0,recall=tp/gtcount,primary_recall=primary_recall,complete_primary=complete,primary_ocr_exact=exact,primary_ocr_exact_q98=strong,negative_windows=2,negative_predictions=negpred,primary_wrong_ocr=bad_primary,detector_ms=stats([r['methods'][method]['detector_ms'] for r in results]),ocr_ms=stats([r['methods'][method]['ocr_ms'] for r in results]))
transfer_summary={}
for method in ['yolo_bus','baseline_production_candidates']:
 hits=0;complete=0;exact=0;strong=0;wrong=[]
 for r in transfers:
  s=r['sample']
  bs=r['boxes'] if method=='yolo_bus' else [dict(box=p['text_box'],ocr=dict(text=p['text']),quality=token_quality(p['token_scores']),gt_metrics=metrics(p['text_box'],s['box_original'])) for p in r['baseline_boxes']]
  primary=[b for b in bs if b['gt_metrics']['iou']>=.5];hits+=bool(primary);complete+=any(b['gt_metrics']['coverage']>=.95 and b['gt_metrics']['area_ratio']<=2 for b in bs)
  exact+=any(exact_route(b['ocr']['text'],s['visible_text']) for b in primary);strong+=any(exact_route(b['ocr']['text'],s['visible_text']) and b['quality']>=.98 for b in primary)
  for b in primary:
   if not exact_route(b['ocr']['text'],s['visible_text']):wrong.append(dict(key=s['key'],text=b['ocr']['text'],quality=b['quality']))
 transfer_summary[method]=dict(primary_recall=hits,complete_primary=complete,primary_ocr_exact=exact,primary_ocr_exact_q98=strong,wrong_ocr=wrong)
transfer_summary['timing']=dict(detector_ms=stats([r['detector_ms'] for r in transfers]),ocr_ms=stats([r['ocr_ms'] for r in transfers]))
result=dict(summary=summary,automatic_bus_transfer_summary=transfer_summary,local_results=results,bus_transfer=transfers,weights_sha256=hashlib.sha256((O/'train_run/weights/best.pt').read_bytes()).hexdigest(),note='Static crops; no tracking/plate filter/freshness decisions. Manual windows are not automatic video accuracy; bus ROI recall has incomplete full-frame GT, no precision claim.')
(O/'detector_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(summary=summary,transfer=transfer_summary),ensure_ascii=False,indent=2))
