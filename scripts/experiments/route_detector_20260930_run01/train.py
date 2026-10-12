"""Frozen small-data dedicated route-display detector experiment."""
import os,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/route_detector_20260930_run01'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')
if (O/'train_run').exists():
    raise FileExistsError('Preserve the existing training run; choose a new experiment directory.')
from ultralytics import YOLO
p=json.loads((O/'protocol.json').read_text());start=time.time()
model=YOLO(str(ROOT/'.cache/yolo11n.pt'))
model.train(data=str(O/'dataset_reviewed/data.yaml'),project=str(O),name='train_run',exist_ok=False,epochs=p['epochs'],imgsz=p['imgsz'],batch=p['batch'],freeze=p['freeze'],seed=p['seed'],deterministic=True,device=0,workers=2,amp=False,optimizer='AdamW',lr0=.001,weight_decay=.0005,patience=60,fliplr=0.,flipud=0.,degrees=10.,translate=.1,scale=.3,mosaic=.5,close_mosaic=10,mixup=0.,copy_paste=0.,plots=True,save=True,verbose=False)
(O/'training_provenance.json').write_text(json.dumps(dict(seconds=time.time()-start,initial_weights_sha256=hashlib.sha256((ROOT/'.cache/yolo11n.pt').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((O/'protocol.json').read_bytes()).hexdigest(),best_weights_sha256=hashlib.sha256((O/'train_run/weights/best.pt').read_bytes()).hexdigest()),indent=2))
