"""Paired original-frame automatic-bus ROI comparison. No targets enter inference."""
import os,json,time,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01';OLD=ROOT/'output/route_detector_20260930_run01';D=O/'dataset_reviewed'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')
from ultralytics import YOLO
from bus_route.recognizer import PARSeqRecognizer
from bus_route.detector import clip_box
from bus_route.target import exact_route,token_quality

def overlap(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1]);return dict(iou=inter/(aa+bb-inter) if aa+bb>inter else 0,coverage=inter/bb if bb else 0,area_ratio=aa/bb if bb else 0)
def stats(x):return dict(p50=float(np.percentile(x,50)),p95=float(np.percentile(x,95)),max=float(max(x))) if x else None
ss=json.loads((D/'manifest.json').read_text());records={r['key']:r for r in json.loads((OLD/'proposals.json').read_text())};samples=[s for s in ss if s['split']=='test' and s['kind']=='bus'];neg=[s for s in ss if s['kind']=='negative']
ocr=PARSeqRecognizer();results={};(O/'evidence').mkdir()
for name,weight,size in [('previous640',OLD/'train_run/weights/best.pt',640),('previous960',OLD/'train_run/weights/best.pt',960),('finetuned640',O/'train/weights/best.pt',640),('finetuned960',O/'train/weights/best.pt',960)]:
 model=YOLO(str(weight))
 for _ in range(3):model.predict(np.zeros((640,640,3),np.uint8),device=0,imgsz=size,rect=False,verbose=False)
 rows=[]
 for s in samples+neg:
  if s['kind']=='bus':
   record=records[s['key']];im=Image.open(OLD/record['frame_path']).convert('RGB');rois=[b['bus_box'] for b in record['buses']];gts=[[a+s['roi'][0],b+s['roi'][1],c+s['roi'][0],d+s['roi'][1]] for a,b,c,d in s['boxes']]
  else:im=Image.open(D/s['image']).convert('RGB');rois=[[0,0,im.width,im.height]];gts=[]
  torch.cuda.synchronize();start=time.perf_counter();boxes=[]
  for roi in rois:
   pred=model.predict(np.array(im.crop(roi))[:,:,::-1].copy(),device=0,imgsz=size,rect=False,conf=.25,iou=.5,verbose=False)[0]
   for b,c in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist()):
    a,y,z,w=clip_box(b,roi[2]-roi[0],roi[3]-roi[1]);boxes.append(dict(box=[a+roi[0],y+roi[1],z+roi[0],w+roi[1]],confidence=c))
  torch.cuda.synchronize();detms=(time.perf_counter()-start)*1000
  preds=ocr.predict_batch([im.crop(b['box']) for b in boxes])
  for b,p in zip(boxes,preds):b.update(ocr=asdict(p),quality=token_quality(p.token_scores),gt=[overlap(b['box'],g) for g in gts])
  prim=[b for b in boxes if b['gt'] and b['gt'][0]['iou']>=.5]
  row=dict(key=s['key'],split=s['split'],kind=s['kind'],visible_text=s['visible_text'],baseline_failed=s.get('baseline_failed'),boxes=boxes,detector_ms=detms,ocr_ms=sum(p.elapsed_ms for p in preds),primary_detected=bool(prim),primary_exact=any(exact_route(b['ocr']['text'],s['visible_text']) for b in prim),primary_strong=any(exact_route(b['ocr']['text'],s['visible_text']) and b['quality']>=.98 for b in prim),primary_complete=any(b['gt'] and b['gt'][0]['coverage']>=.95 and b['gt'][0]['area_ratio']<=2 for b in boxes))
  rows.append(row)
  if name=='finetuned960':
   view=im.copy();draw=ImageDraw.Draw(view)
   for gt in gts:draw.rectangle(gt,outline='lime',width=3)
   for b in boxes:draw.rectangle(b['box'],outline='red',width=3)
   view.thumbnail((720,900));view.save(O/'evidence'/f"{s['key']}.jpg",quality=95)
   for j,b in enumerate(boxes):im.crop(b['box']).save(O/'evidence'/f"{s['key']}_{j}.png")
  print(name,s['key'],row['primary_exact'],flush=True)
 pos=[r for r in rows if r['kind']=='bus'];ns=[r for r in rows if r['kind']=='negative'];summary=dict(test_positive_frames=len(pos),**{k:sum(r[k] for r in pos) for k in ['primary_detected','primary_exact','primary_strong','primary_complete']},normal_exact=sum(r['primary_exact'] for r in pos if not r['baseline_failed']),failure_exact=sum(r['primary_exact'] for r in pos if r['baseline_failed']),negative_candidates={r['key']:len(r['boxes']) for r in ns},high_conf_numeric_negatives=[dict(key=r['key'],text=b['ocr']['text'],quality=b['quality']) for r in ns for b in r['boxes'] if any(c.isdigit() for c in b['ocr']['text']) and b['quality']>=.98],detector_ms=stats([r['detector_ms'] for r in pos]),ocr_ms=stats([r['ocr_ms'] for r in pos]))
 results[name]=dict(summary=summary,rows=rows,weights_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),imgsz=size)
 (O/f'{name}_evaluation.json').write_text(json.dumps(results[name],ensure_ascii=False,indent=2));del model
(O/'comparison.json').write_text(json.dumps({k:v['summary'] for k,v in results.items()},ensure_ascii=False,indent=2));print(json.dumps({k:v['summary'] for k,v in results.items()},ensure_ascii=False,indent=2))
