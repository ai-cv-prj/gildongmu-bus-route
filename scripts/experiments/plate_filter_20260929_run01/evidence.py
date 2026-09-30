"""Keep corrected source attribution, protected PRINT anchors, and changed decisions."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw
from bus_route.target import exact_route, token_quality
from bus_route.video import sample_frames
from evaluate import ROOT, OUT, BASE, load


def main():
    dest=OUT/'evidence';dest.mkdir(exist_ok=True)
    records=json.loads((BASE/'evidence/visual_review.json').read_text())
    inputs=list(csv.DictReader((OUT/'inputs.csv').open(encoding='utf-8-sig')))
    fixtures=json.loads((ROOT/'scripts/tests/fixtures/route_evidence/cases.json').read_text())
    for case in fixtures:
        if case['id']<26:continue
        index=next(i for i,r in enumerate(inputs,1) if r['video_path']==case['video'])
        folder=f'video_{index:02d}'
        data=load(ROOT/'output/hold_fix_20260929_run01'/folder)
        p=max((p for p in data[1] if float(p['timestamp_s'])==case['timestamp_s']
              and p['text']==case['text']
              and p['text_box']==case['source_box']),
              key=lambda p:token_quality(p['token_scores']))
        records.append({'id':len(records),'folder':folder,'target':case['text'],
            'video_path':case['video'],'first_s':case['timestamp_s'],'box':p['text_box'],
            'ocr_text':p['text'],'score':token_quality(p['token_scores']),
            'crop':f'{len(records):02d}_crop.png','context':f'{len(records):02d}_context.jpg',
            'kind':'regression_guard','verdict':'route_display',
            'review_note':('녹색 차체 위 밝은 측면 PRINT 03.' if case['text']=='03'
                           else '어두운 바탕의 밝은 측면 LED 6713.')
                          +' 초기 색상 확장 필터의 회귀를 밝은 바탕 조건으로 수정한 보호 사례.'})
    for r in records:
        if r['id']<26:
            for key in ['crop','context']:
                shutil.copyfile(BASE/'evidence'/r[key],dest/r[key])
        if r['id']==24:
            r['review_note']='뒤쪽 7611 버스 광고의 전화번호 1577-1987 앞 4자리. 전면 번호판이라는 이전 설명을 원본 좌표 대조 후 정정.'
        if r['id']==25:
            r['review_note']='뒤쪽 7611 버스 후면의 그림자 속 등록번호판 9350. 별도 차체 식별번호라는 이전 설명을 정정.'
        for mode in ['offline','realtime']:
            data=load(OUT/mode/r['folder'])
            ps=[p for p in data[1] if float(p['timestamp_s'])==r['first_s']
                and exact_route(p['text'],r['target']) and p['text_box']==r['box']]
            r[mode+'_anchor']=[{k:p[k] for k in ['text','text_box','bus_box','eligible','rejection_reason','context_checks']} for p in ps]
    changes=[]
    for mode in ['offline','realtime']:
        summary=json.loads((OUT/f'{mode}_evaluation.json').read_text())
        for r in summary['gains']:
            changes.append({**r,'mode':mode,'kind':'gain'})
        for r in summary['new_alternative_flags']:
            changes.append({**r,'mode':mode,'kind':'new_alternative'})
        for r in summary['losses']:
            old=next(x for x in csv.DictReader((BASE/'comparison.csv').open(encoding='utf-8-sig')) if x['folder']==r['folder'])
            # Preserve the actual old successful crop for a lost video.
            changes.append({**r,'mode':mode,'kind':'loss','first_s':float(old['first_recognized_s'])})
    grouped=defaultdict(list)
    for r in changes:
        data=load((BASE/'realtime' if r['kind']=='loss' else OUT/r['mode'])/r['folder'])
        ps=[p for p in data[1] if abs(float(p['timestamp_s'])-r['first_s'])<1e-6
            and p['eligible'] and exact_route(p['text'],r['target'])]
        if r.get('track_id') is not None:
            ps=[p for p in ps if str(r['track_id'])==p['track_id']]
        if not ps:raise RuntimeError(f'missing evidence: {r}')
        p=max(ps,key=lambda p:token_quality(p['token_scores']))
        index=len(records)
        row={**r,'id':index,'box':p['text_box'],'ocr_text':p['text'],
             'score':token_quality(p['token_scores']),'crop':f'{index:02d}_crop.png',
             'context':f'{index:02d}_context.jpg','verdict':'needs_review','review_note':''}
        records.append(row);grouped[r['video_path']].append(row)
    # Detailed original attribution of the three reported source mistakes.
    for r in records:
        if r['id'] in [19,24,25] or r['kind']=='regression_guard':
            grouped[r['video_path']].append(r)
    for path,rows in grouped.items():
        reader=sample_frames(ROOT/path,5);next(reader)
        try:
            for t,rgb in reader:
                for r in rows:
                    if abs(t-r['first_s'])>1e-6:continue
                    im=Image.fromarray(rgb)
                    if r['id']>=26:
                        im.crop(r['box']).save(dest/r['crop'])
                        draw=ImageDraw.Draw(im);draw.rectangle(r['box'],outline='red',width=4)
                        im.thumbnail((720,960));im.save(dest/r['context'],quality=95)
                    else:
                        ps=r['offline_anchor']
                        if not ps: continue
                        p=min(ps,key=lambda p:sum(abs(a-b) for a,b in zip(p['text_box'],r['box'])))
                        draw=ImageDraw.Draw(im);draw.rectangle(p['bus_box'],outline='orange',width=3)
                        draw.rectangle(p['text_box'],outline='red',width=3)
                        a,b,c,d=p['bus_box'];pad=round((d-b)*.15)
                        im=im.crop((max(0,a-pad),max(0,b-pad),min(im.width,c+pad),min(im.height,d+pad)))
                        im.thumbnail((850,850));im.save(dest/f'{r["id"]:02d}_attribution.jpg',quality=95)
                        r['attribution']=f'{r["id"]:02d}_attribution.jpg'
                        for j,ctx in enumerate(p['context_checks']):
                            Image.fromarray(rgb).crop(ctx['box']).save(dest/f'{r["id"]:02d}_context_ocr_{j}.png')
                if t>=max(r['first_s'] for r in rows): break
        finally:reader.close()
    (dest/'review.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    for start in range(26,len(records),6):
        sheet=Image.new('RGB',(1440,1240),'#eee');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(records[start:start+6]):
            x=j%3*480;y=j//3*620
            draw.text((x+5,y+5),f'{r["id"]} {r["kind"]} {r["folder"]} {r["target"]} t={r["first_s"]}',fill='black')
            im=Image.open(dest/r['context']);im.thumbnail((465,460));sheet.paste(im,(x+5,y+30))
            im=Image.open(dest/r['crop']);im.thumbnail((465,110));sheet.paste(im,(x+5,y+505))
        sheet.save(dest/f'new_sheet_{start:02d}.jpg',quality=95)
    print('evidence',len(records),'new',len(changes),flush=True)


if __name__=='__main__':main()
