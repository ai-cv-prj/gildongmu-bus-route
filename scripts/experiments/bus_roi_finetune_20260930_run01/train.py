import os,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01';p=json.loads((O/'protocol.json').read_text())
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')
from ultralytics import YOLO
assert not (O/'train').exists()
start=time.time();model=YOLO(str(ROOT/p['initial_weights']))
model.train(data=str(O/'dataset_reviewed/data.yaml'),project=str(O),name='train',epochs=p['epochs'],imgsz=p['imgsz'],batch=p['batch'],freeze=p['freeze'],lr0=p['lr0'],optimizer='AdamW',weight_decay=.0005,device=0,workers=2,amp=False,seed=p['seed'],deterministic=True,patience=100,fliplr=0,flipud=0,degrees=10,translate=.15,scale=.5,mosaic=.5,close_mosaic=15,mixup=0,copy_paste=0,plots=True,verbose=False)
(O/'training_provenance.json').write_text(json.dumps(dict(seconds=time.time()-start,initial_sha256=hashlib.sha256((ROOT/p['initial_weights']).read_bytes()).hexdigest(),best_sha256=hashlib.sha256((O/'train/weights/best.pt').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((O/'protocol.json').read_bytes()).hexdigest()),indent=2))
