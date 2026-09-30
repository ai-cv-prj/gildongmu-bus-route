"""Manual full-number crops: diagnostic only, never end-to-end evaluation."""
import json
from pathlib import Path
from PIL import Image
from bus_route.recognizer import PARSeqRecognizer
root=Path(__file__).resolve().parents[3]
out=root/'output/diagnosis_20260929_run01'
specs=[(24,'0.0_extra',(844,100,889,121),'273'),(28,'2.0_frame',(783,50,900,91),'7011'),(29,'9.8_frame',(925,765,1000,865),'M7731'),(40,'6.0_frame',(840,535,1025,715),'M6628'),(41,'6.0_frame',(840,535,1025,715),'M6628'),(62,'1.2_frame',(400,130,640,255),'602')]
model=PARSeqRecognizer('cuda');rows=[]
for n,frame,box,target in specs:
 source=f'video_{n:02d}_{frame}.png';name=f'video_{n:02d}_manual_full.png';im=Image.open(out/source).crop(box);im.save(out/name);p=model.predict(im)
 row=dict(folder=f'video_{n:02d}',source=source,box=box,target=target,crop=name,text=p.text,min_token_score=min(p.token_scores),token_scores=p.token_scores,diagnostic_only=True);rows.append(row);print(row,flush=True)
(out/'manual_crop_probe.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(out/'probe_model.json').write_text(json.dumps(model.metadata(),ensure_ascii=False,indent=2))
