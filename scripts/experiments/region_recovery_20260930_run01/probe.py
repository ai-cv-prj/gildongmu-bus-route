"""Diagnostic detector/OCR probes, never counted as full-video evaluation."""
import json
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from bus_route.detector import BusTextDetector
from bus_route.recognizer import PARSeqRecognizer
ROOT=Path(__file__).resolve().parents[3]
out=ROOT/'output/region_recovery_20260930_run01'
torch.set_num_threads(4)
d=BusTextDetector();r=PARSeqRecognizer()
rows=[]
for n,src,search,box in [(102,'video_28_2.0_frame.png',(700,12,942,107),(790,53,844,84)),(114,'video_40_6.0_frame.png',(494,281,1080,1081),(893,556,1017,707)),(103,'video_29_9.8_frame.png',(787,701,1080,1119),(960,821,998,855))]:
 im=Image.open(ROOT/'output/diagnosis_20260929_run01'/src).convert('RGB');rgb=np.array(im)
 x,y,z,w=search
 roi=rgb[y:w,x:z];resized,ratio,_=d.imgproc.resize_aspect_ratio(roi,1280,interpolation=cv2.INTER_LINEAR,mag_ratio=1.5)
 tensor=torch.from_numpy(d.imgproc.normalizeMeanVariance(resized)).permute(2,0,1)
 with torch.inference_mode(): score=d.craft(tensor[None].to('cuda'))[0][0].cpu().numpy()
 for low in [.4,.25,.15]:
  boxes,_=d.utils.getDetBoxes(score[:,:,0],score[:,:,1],.5,.2,low,False)
  boxes=d.utils.adjustResultCoordinates(boxes,1/ratio,1/ratio)
  print(n,low,[np.round(b+[x,y]).astype(int).tolist() for b in boxes],flush=True)
 canvas=im.crop((max(0,box[0]-100),max(0,box[1]-50),min(im.width,box[2]+100),min(im.height,box[3]+50)))
 canvas.save(out/f'probe_{n}_context.png')
 a,b,c,e=box;h=e-b
 specs=[]
 for left,right in [(0,1.5),(.5,0),(1,1),(.5,1.5),(.35,.35)]:
  cropbox=(max(x,round(a-left*h)),max(y,b-3),min(z,round(c+right*h)),min(w,e+3))
  specs.append((left,right,cropbox))
 for spec,p in zip(specs,r.predict_batch([im.crop(s[2]) for s in specs])):
  row=dict(video=n,left=spec[0],right=spec[1],box=spec[2],text=p.text,score=min(p.token_scores));rows.append(row);print(row,flush=True)
  im.crop(spec[2]).save(out/f'probe_{n}_{spec[0]}_{spec[1]}.png')
(out/'probe.json').write_text(json.dumps(rows,indent=2))
