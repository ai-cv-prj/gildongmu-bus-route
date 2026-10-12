import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
O=Path('output/route_detector_20260930_run01');ls=json.loads((O/'manual_draft.json').read_text());rs=json.loads((O/'proposals.json').read_text())
if (O/'manual_labels.json').exists():
 raise FileExistsError('Preserve reviewed labels; choose a new experiment directory.')
updates={14:([734,456,808,521],None),51:([148,501,540,827],None),59:([464,665,529,703],None),50:([775,738,928,842],'v050_012')}
for l in ls:
 if l['video_index'] in updates:
  b,k=updates[l['video_index']];l['bbox']=b
  if k:l['key']=k
 if l['video_index'] in [14,55,59,76,98,130]:l['visibility']='uncertain'
 if l['video_index']==108:l['visibility']='other_visible_route'
 if l['video_index']==39:l['bbox']=[471,933,508,990]
 if l['video_index']==97:l['bbox']=[566,686,619,786]
 l['reviewer']='Codex visual review; no new OCR consulted';l['review_status']='reviewed'
# Quadrilaterals TL,TR,BR,BL are optional diagnostic geometry, no OCR-based search.
quads={47:[[29,570],[155,642],[137,677],[13,606]],48:[[96,651],[207,718],[188,754],[77,689]],51:[[155,713],[482,502],[535,610],[199,826]],103:[[939,783],[999,770],[1000,859],[935,862]],24:[[1048,642],[1105,665],[1104,701],[1045,681]],21:[[997,320],[1119,296],[1120,397],[993,400]]}
for l in ls:
 if l['video_index'] in quads:l['quad']=quads[l['video_index']]
(O/'manual_labels.json').write_text(json.dumps(ls,ensure_ascii=False,indent=2))
print(hashlib.sha256((O/'manual_labels.json').read_bytes()).hexdigest())
l=next(x for x in ls if x['video_index']==50);rec=next(x for x in rs if x['key']==l['key']);im=Image.open(O/rec['frame_path']).crop(l['bbox']);im.resize((459,312)).save(O/'sheets/50_check.png')
