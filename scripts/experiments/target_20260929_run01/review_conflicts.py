"""원시 예측의 실제 crop/문맥 복원. 모델 재추론이나 라벨 변경 없음."""
import csv
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import imageio_ffmpeg
from bus_route.detector import ROOT
from bus_route.target import exact_route
from bus_route.video import sample_frames,color_filters


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    run=ROOT/'output/target_20260929_run01';out=run/'conflict_review'
    out.mkdir(exist_ok=False)
    items=[]
    for row in read(run/'target_video_summary.csv'):
        if not(row['test_kind']=='filename_target_simulation' and row['raw_match']=='True' and row['rule_match']=='False' and row['conflict_seen']=='True'):continue
        folder=run/row['folder'];preds=read(folder/'predictions.csv');decisions=read(folder/'decisions.csv');chosen=[];seen=set()
        for decision in decisions:
            for text in json.loads(decision['conflicts']):
                key=(decision['track_id'],text)
                if key in seen:continue
                seen.add(key)
                possible=[p for p in preds if p['track_id']==key[0] and p['text']==text and p['eligible']=='True'
                          and min(json.loads(p['token_scores']) or [0])>=.75 and float(p['timestamp_s'])<=float(decision['timestamp_s'])+1e-6]
                assert possible
                chosen.append(('conflict',possible[0]))
        # 각 목표 일치 트랙의 최초 예측도 같은 문맥으로 제시.
        target_tracks=set()
        for pred in preds:
            if exact_route(pred['text'],row['target_route']) and pred['track_id'] not in target_tracks:
                target_tracks.add(pred['track_id']);chosen.append(('target',pred))
        by_time={}
        for kind,pred in chosen:by_time.setdefault(float(pred['timestamp_s']),[]).append((kind,pred))
        reader=sample_frames(ROOT/row['video_path'],5);next(reader)
        for ts,rgb in reader:
            if ts not in by_time:continue
            for kind,pred in by_time[ts]:
                uid=f'{row["folder"]}_{kind}_{len(items)+1:02d}'
                box=json.loads(pred['text_box']);image=Image.fromarray(rgb);crop=image.crop(box);crop.save(out/f'{uid}_crop.png')
                draw=ImageDraw.Draw(image);draw.rectangle(box,outline='red',width=5)
                image.thumbnail((540,960));image.save(out/f'{uid}_context.jpg',quality=90)
                items.append({'id':uid,'video_path':row['video_path'],'folder':row['folder'],'target_route':row['target_route'],
                              'kind':kind,'timestamp_s':ts,'track_id':pred['track_id'],'text':pred['text'],
                              'box':box,'search_zone':pred['search_zone'],'min_token_score':min(json.loads(pred['token_scores']) or [0]),
                              'context_image':f'{uid}_context.jpg','crop_image':f'{uid}_crop.png'})
    for start in range(0,len(items),8):
        sheet=Image.new('RGB',(1440,1440),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(items[start:start+8]):
            x=(j%4)*360;y=(j//4)*720
            draw.text((x+5,y+5),f'{item["id"]} t={item["timestamp_s"]:.1f} ID={item["track_id"]}',fill='black')
            draw.text((x+5,y+25),f'OCR={item["text"]} zone={item["search_zone"]} score={item["min_token_score"]:.3f}',fill='black')
            im=Image.open(out/item['context_image']);im.thumbnail((350,530));sheet.paste(im,(x+5,y+50))
            crop=Image.open(out/item['crop_image']);scale=min(330/crop.width,110/crop.height,6)
            crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))))
            sheet.paste(crop,(x+5,y+590))
        sheet.save(out/f'sheet_{start//8+1:02d}.jpg',quality=93)
    for index in [18,23]:
        name=read(run/'inputs.csv')[index-1]['video_path'];filters,_=color_filters(ROOT/name)
        reader=imageio_ffmpeg.read_frames(str(ROOT/name),output_params=['-vf',filters,'-threads','2']);meta=next(reader);last=None;count=0
        for frame in reader:last=frame;count+=1
        w,h=meta['size'];im=Image.fromarray(np.frombuffer(last,np.uint8).reshape(h,w,3));im.save(out/f'video_{index:02d}_last_frame.png')
    (out/'evidence.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
    print('evidence',len(items),'sheets',(len(items)+7)//8)


if __name__=='__main__':main()
