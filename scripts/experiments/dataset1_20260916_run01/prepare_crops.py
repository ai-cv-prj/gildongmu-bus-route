# 파일명: prepare_crops.py
# 설명: 고정 중간 프레임에서 육안 지정한 LED 영역을 원본 해상도로 추출한다.
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/dataset1_20260916_run01'
# 검토 시트의 각 프레임(1/3 축소본) 기준 좌표. 모델 예측을 보기 전에 지정.
BOXES = {
 1:(20,335,55,348),2:(128,320,141,329),3:(126,205,143,216),4:(115,140,141,158),
 6:(53,42,78,62),7:(282,57,304,71),8:(53,0,119,20),9:(220,0,267,34),10:(193,28,277,59),
 11:(253,105,289,135),12:(201,200,245,221),13:(411,15,454,49),14:(343,91,377,124),
 15:(279,111,322,139),16:(307,134,350,152),17:(326,105,358,126),18:(380,78,442,107),
 20:(382,98,401,116),21:(169,80,269,96),22:(147,85,180,101),23:(191,102,229,119),24:(230,47,256,63),
 26:(84,305,113,321),27:(159,222,177,245),28:(85,254,137,276),29:(170,247,189,278),
 30:(84,276,101,290),31:(121,273,147,292),32:(258,209,310,248),33:(84,250,104,265),
 35:(256,61,294,99),38:(156,193,227,231),39:(13,187,37,211),40:(9,212,30,231),
 41:(60,261,117,290),42:(92,206,150,252),43:(16,191,134,289),44:(225,260,304,280),
 46:(23,263,36,271),47:(231,150,279,175),48:(44,322,77,337),49:(112,103,185,126),
 50:(105,133,148,153),51:(80,152,104,168),52:(131,225,147,239),53:(25,192,55,205),
 54:(74,161,168,190),55:(299,108,350,138),56:(139,114,181,132),57:(121,162,147,180),
 58:(135,337,156,351),59:(40,234,58,249),60:(22,164,58,185),61:(34,239,58,254),62:(67,302,94,315),
}
SKIPS = {5:'target front outside frame; side number not identifiable',19:'target panel occluded',25:'front panel outside frame',34:'visible panel shows message; route digits not identifiable',36:'visible 1000 is printed; no identifiable LED number',37:'front number absent/clipped; no identifiable LED number',45:'foreground target shows printed side number only',63:'target obscured by foreground buses'}
SIDE = {27,29,35,43}
rows=json.loads((OUT/'manifest.json').read_text())
crop_dir=ROOT/'data/crops/dataset1'
crop_dir.mkdir(exist_ok=True)
annotations=[]
for r in rows:
    i=r['index']
    r['sample_id']=f'led_{i:02d}'
    r['review_status']='crop_pending_review' if i in BOXES else 'no_led_number_roi_at_midpoint'
    r['note']=SKIPS.get(i,'')
    if i not in BOXES: continue
    box=tuple(v*3 for v in BOXES[i])
    path=crop_dir/f"led_{i:02d}.png"
    with Image.open(r['frame_path']) as im:
        im.crop(box).save(path)
    r.update(bbox_xyxy=box,crop_path=str(path.relative_to(ROOT)),view='side' if i in SIDE else 'front')
    annotations.append(dict(sample_id=r['sample_id'],video_id=r['video_id'],event_id=f"review_video_{i:02d}",
        timestamp_s=r['timestamp_s'],split='dev',route_id_gt=r['route_id_gt'],
        visible_text_gt=r['route_id_gt'].removeprefix('서대문'),x1=box[0],y1=box[1],x2=box[2],y2=box[3],
        visibility='unverified',display_type='LED',gt_source='user filename; pending manual crop review',
        crop_path=r['crop_path'],color_mode=r['color_mode'],view=r['view']))
(OUT/'review.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
with (OUT/'annotations.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(annotations[0])); writer.writeheader();writer.writerows(annotations)
for start in range(0,len(annotations),15):
    subset=annotations[start:start+15]
    sheet=Image.new('RGB',(1200,5*180),'#eeeeee');draw=ImageDraw.Draw(sheet)
    for j,a in enumerate(subset):
        x=(j%3)*400;y=(j//3)*180
        with Image.open(ROOT/a['crop_path']) as im:
            im.thumbnail((380,130));sheet.paste(im,(x+5,y+35))
        draw.text((x+5,y+5),f"{a['sample_id']} / expected {a['visible_text_gt']} / {a['view']}",fill='black')
    sheet.save(OUT/'contact_sheets'/f'crops_{start//15:02d}.jpg')
print('crops',len(annotations),'no ROI',len(SKIPS))
