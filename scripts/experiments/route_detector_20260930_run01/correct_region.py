"""Visual review found v64 reference box clipped the top. Preserve initial diagnosis."""
import json
from pathlib import Path
from PIL import Image
from bus_route.video import sample_frames
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/route_detector_20260930_run01';r=next(r for r in json.loads((O/'proposals.json').read_text()) if r['key']=='v064_015');b=[539,409,589,455]
if (O/'manual_label_correction.json').exists():
    raise FileExistsError('Preserve the existing correction; choose a new experiment directory.')
reader=sample_frames(ROOT/r['video_path'],5);next(reader)
try:
 for t,arr in reader:
  if t==3.:break
finally:reader.close()
im=Image.fromarray(arr)
for name,box in [('exact',b),('pad10',[534,404,594,460])]:im.crop(box).save(O/'manual_crops'/f'v064_015_review2_{name}.png')
(O/'manual_label_correction.json').write_text(json.dumps(dict(key=r['key'],bbox=b,reason='Second visual context check shows digits extend above reference box. Coordinates chosen visually, not OCR score.',variants=['exact','pad10']),indent=2))
