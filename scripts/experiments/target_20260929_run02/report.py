"""수정 후 재평가: 전체 영상 비교·다른 목표 재생·첫 판단 crop 검토 자료."""
import csv
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from bus_route.detector import ROOT
from bus_route.target import TargetMatcher,exact_route,route_aliases
from bus_route.video import sample_frames

OLD=ROOT/'output/target_20260929_run01'
OUT=ROOT/'output/target_20260929_run02'
WORK=ROOT/'workbench/target_20260929_run02'

def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def dump(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def csvsave(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def predictions(folder):
    preds=read(folder/'predictions.csv')
    for p in preds:
        p['token_scores']=json.loads(p['token_scores']);p['eligible']=p['eligible']=='True'
        p['text_box']=json.loads(p['text_box'])
    return preds

def replay(frames,preds,target,alternatives=()):
    by_frame={}
    for p in preds:by_frame.setdefault(p['sample_index'],[]).append(p)
    m=TargetMatcher(target,alternatives);matches=[]
    for f in frames:
        for j,b in enumerate(json.loads(f['buses'])):
            observations=[p for p in by_frame.get(f['sample_index'],[]) if int(p['bus_index'])==j]
            d=m.update(b['track_id'],float(f['timestamp_s']),observations,b['bus_box'],not b['candidate_truncated'])
            if d['state']=='matched_candidate':matches.append({'timestamp_s':float(f['timestamp_s']),'track_id':b['track_id']})
    return matches

def alternate_targets(route,known,preds):
    # 실험용 다른 목표. 실제 API 도착 후보라고 가정하지 않는다.
    key=re.sub(r'^[가-힣]+','',route);targets=set(known)|{'7016','7017'}
    for i,c in enumerate(key):
        if c.isdigit():
            for offset in [-1,1]:targets.add(key[:i]+str((int(c)+offset)%10)+key[i+1:])
    for p in preds:
        text=p['text']
        if re.fullmatch(r'(?:M|N)?\d+[AB]?(?:-\d+)?',text) and len(text)==len(key):targets.add(text)
    return sorted(t for t in targets if not (route_aliases(t) & route_aliases(route)))

def main():
    WORK.mkdir(exist_ok=True); audit=OUT/'candidate_audit';audit.mkdir(exist_ok=True)
    inputs=read(OLD/'inputs.csv');old=read(OLD/'target_video_summary.csv');sums=json.loads((OUT/'summary.json').read_text())
    assert len(inputs)==len(sums)==71
    known=sorted({r['route_hint'] for r in inputs});rows=[];times=[];counter=[];rejects=Counter();validation=[];audititems=[]
    for i,(inp,prev,s) in enumerate(zip(inputs,old,sums),1):
        folder=OUT/f'video_{i:02d}';frames=read(folder/'frames.csv');preds=predictions(folder);decisions=read(folder/'decisions.csv')
        reader=imageio_ffmpeg.read_frames(str(folder/'annotated.mp4'));next(reader)
        n=sum(1 for _ in reader);reader.close()
        assert n==s['sample_count']==len(frames) and len(preds)==s['prediction_count']
        assert s['video']==inp['video_path'] and s['target_route']==inp['target_route']
        w,h=s['metadata']['size']
        for p in preds:
            a,b,c,d=p['text_box'];assert 0<=a<c<=w and 0<=b<d<=h
        times.extend(float(f['pipeline_ms']) for f in frames)
        rejects.update(p['rejection_reason'] for p in preds if p['rejection_reason'])
        own=replay(frames,preds,inp['target_route'])
        assert bool(own)==(s['first_matched_candidate_s'] is not None)
        # 이전 OCR에 새 시간 판단만 적용한 참고값. crop 필터·주기 효과와 구분한다.
        old_policy=replay(read(OLD/f'video_{i:02d}'/'frames.csv'),predictions(OLD/f'video_{i:02d}'),inp['target_route'])
        row={**inp,'folder':folder.name,'old_raw':prev['raw_match']=='True','old_match':prev['rule_match']=='True',
             'policy_only_match':bool(old_policy),'new_raw':s['target_match_samples']>0,'new_match':bool(own),
             'eligible_exact_frames':len({p['timestamp_s'] for p in preds if p['eligible'] and exact_route(p['text'],inp['target_route'])}),
             'first_match_s':s['first_matched_candidate_s'],'raw_match_samples':s['target_match_samples'],
             'pipeline_p95_ms':s['pipeline_ms']['p95'],'conflict_frames':s['decision_counts'].get('conflict',0)}
        rows.append(row)
        validation.append({'folder':folder.name,'frames':n,'predictions':len(preds),'coordinates':'pass','replayed_decisions':'pass'})
        for target in alternate_targets(inp['route_hint'],known,preds):
            hits=replay(frames,preds,target)
            counter.append({'folder':folder.name,'video_path':inp['video_path'],'filename_route':inp['route_hint'],
                            'target_route':target,'matched_frames':len(hits),'first_s':hits[0]['timestamp_s'] if hits else None,
                            'track_id':hits[0]['track_id'] if hits else None})
        if own:
            hit=own[0];ts=hit['timestamp_s']
            matches=[p for p in preds if float(p['timestamp_s'])==ts and int(p['track_id'])==hit['track_id'] and p['eligible'] and exact_route(p['text'],inp['target_route']) and min(p['token_scores'])>=.9]
            assert matches
            best=max(matches,key=lambda p:min(p['token_scores']))
            reader=sample_frames(ROOT/inp['video_path'],5);next(reader)
            try:
                for timestamp,rgb in reader:
                    if timestamp==ts:
                        im=Image.fromarray(rgb);im.crop(best['text_box']).save(audit/f'{folder.name}_crop.png')
                        ImageDraw.Draw(im).rectangle(best['text_box'],outline='red',width=5)
                        im.thumbnail((450,800));im.save(audit/f'{folder.name}_context.jpg',quality=90);break
            finally:reader.close()
            audititems.append({'folder':folder.name,'video_path':inp['video_path'],'timestamp_s':ts,'text':best['text'],'box':best['text_box'],'track_id':hit['track_id']})
        print(f'report {i}/71',flush=True)
    for start in range(0,len(audititems),10):
        sheet=Image.new('RGB',(1500,1400),'#eee');draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(audititems[start:start+10]):
            x=j%5*300;y=j//5*700
            draw.text((x+5,y+5),f'{item["folder"]}: {item["text"]} @ {item["timestamp_s"]:.1f}s',fill='black')
            im=Image.open(audit/f'{item["folder"]}_context.jpg');im.thumbnail((290,530));sheet.paste(im,(x+5,y+30))
            crop=Image.open(audit/f'{item["folder"]}_crop.png');scale=min(280/crop.width,110/crop.height,6)
            crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))))
            sheet.paste(crop,(x+5,y+575))
        sheet.save(audit/f'sheet_{start//10+1:02d}.jpg',quality=94)
    positive=rows[:70]
    totals={k:sum(r[k] for r in positive) for k in ['old_raw','old_match','policy_only_match','new_raw','new_match']}
    totals.update({'positive_videos':70,'negative_7017_target_7016':rows[-1],
                   'gained':[r['video_path'] for r in positive if r['new_match'] and not r['old_match']],
                   'lost':[r['video_path'] for r in positive if r['old_match'] and not r['new_match']],
                   'unmatched':[r['video_path'] for r in positive if not r['new_match']],
                   'pipeline_ms':{'p50':float(np.percentile(times,50)),'p95':float(np.percentile(times,95)),'max':max(times)},
                   'frames':len(times),'rejected_predictions':dict(rejects),'counterfactual_cases':len(counter),
                   'counterfactual_flags':[r for r in counter if r['matched_frames']],
                   'tags':{tag:{'videos':sum(r['display_tag']==tag for r in positive),'matched':sum(r['display_tag']==tag and r['new_match'] for r in positive)} for tag in sorted({r['display_tag'] for r in positive})}})
    realtime=json.loads((OUT/'realtime/summary.json').read_text());ages=[];rtimes=[];rtchecks=[]
    for i,s in enumerate(realtime,1):
        f=OUT/'realtime'/f'video_{i:02d}';frames=read(f/'frames.csv')
        reader=imageio_ffmpeg.read_frames(str(f/'annotated.mp4'));next(reader);n=sum(1 for _ in reader);reader.close()
        assert n==s['rendered_frame_count'] and n-len(frames)==s['dropped_samples']
        ages.extend(float(r['frame_age_ms']) for r in frames);rtimes.extend(float(r['pipeline_ms']) for r in frames)
        rtchecks.append({'video':s['video'],'processed':len(frames),'rendered':n,'dropped':s['dropped_samples'],'matched':s['first_matched_candidate_s'] is not None})
    totals['realtime']={'videos':len(realtime),'processed':len(ages),'dropped':sum(r['dropped'] for r in rtchecks),'frame_age_p95_ms':float(np.percentile(ages,95)),'frame_age_max_ms':max(ages),'pipeline_p95_ms':float(np.percentile(rtimes,95)),'results':rtchecks}
    dump(OUT/'comparison_summary.json',totals);csvsave(OUT/'comparison.csv',rows);csvsave(OUT/'counterfactual.csv',counter)
    dump(OUT/'verification.json',{'offline':validation,'realtime':rtchecks});dump(audit/'evidence.json',audititems)
    shutil.copy2(OLD/'inputs.csv',OUT/'inputs.csv');shutil.copy2(OLD/'replay_inputs.csv',OUT/'replay_inputs.csv')
    sources=[*sorted((ROOT/'core/bus_route').glob('*.py')),*sorted((ROOT/'scripts/tests').glob('*.py')),ROOT/'scripts/infer_target_video.py',Path(__file__).resolve()]
    dump(OUT/'source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    base='../../output/'+OUT.name+'/'
    lines=['# 번호판·잘린 crop·충돌 처리 수정 후 평가','','## 결론',f'목표 번호 확인 조건을 통과한 영상은 **{totals["new_match"]}/70 ({totals["new_match"]/70:.1%})**. 이전 {totals["old_match"]}/70 ({totals["old_match"]/70:.1%}) 대비 {(totals["new_match"]-totals["old_match"])/70*100:.1f}%p 변화. 새로 통과 {len(totals["gained"])}개, 기존 통과에서 보류로 바뀐 영상 {len(totals["lost"])}개.',
           '',f'원시 OCR에서 목표 문자열이 한 번 이상 나온 영상: {totals["old_raw"]}/70 → {totals["new_raw"]}/70. 이전 저장 OCR에 새 시간 판단만 적용하면 {totals["policy_only_match"]}/70이며, 이것은 crop 필터와 탐색 주기를 바꾼 전체 재추론 결과와 다르다.',
           '', '**여기서 수치는 영상별 목표 번호 확인 성공률이다. 모든 출력의 정답률(precision), 프레임별 OCR 정확도 또는 실제 탑승 안전도를 뜻하지 않는다.** 모든 입력에 목표가 등장한다는 사용자 라벨을 사용했다. 이미 오류를 검토한 dataset2를 다시 평가했으므로 독립 검증셋 성능이 아니다.',
           '', '[전체 영상 비교·첫 판단 crop](review.html)','','## 수정 내용',
           '- 원본 모델(YOLO11n / CRAFT / PARSeq)은 유지했다. 학습·목표 번호를 넣은 OCR·숫자 강제 치환은 하지 않았다.',
           '- 아래쪽 노란 바탕 crop을 차량번호판 의심으로 제외한다. 내부 탐색 경계 1px에 닿은 crop과 3문자 이상인데 폭이 지나치게 좁은 출력을 제외한다. 제외 이유도 predictions.csv에 남긴다.',
           '- 차체 탐색 주기 0.6초 → 0.2초. 5fps 전체 분석 프레임에서 측면 PRINT의 짧은 노출을 관찰한다. 버스당 OCR 후보 24개, 배치 16개 제한은 유지한다.',
           '- 같은 길이라는 이유만으로 모두 경쟁 번호로 취급하지 않는다. 명시된 API 후보 또는 한 글자 차이의 지원 형식 번호를 감시한다. A/B 접미사 지원은 이번 실험 설정이며 전국 노선의 종류를 단정하지 않는다.',
           '- 최근 1.2초의 고점수 경쟁 번호(0.9 이상), 반복 경쟁 번호(0.75 이상 2시점), 현재 프레임의 동시 경쟁은 보류한다. 낮은 점수의 단발 오독은 전체 목표 번호 3시점과 구별 문자 점수 0.97 이상으로만 해소한다. 과거 충돌을 영상 끝까지 유지하지 않는다.',
           '- 전체 일치 3시점, 최소 0.4초, 모든 문자와 EOS 점수 0.9 이상을 요구한다. 701 같은 부분 번호는 여전히 7016/7017로 보완하지 않는다. 트랙 혼합·추적 단절·후보 잘림·실시간 지연도 보류한다.',
           '', '**제한:** 이 필터는 학습된 번호판/광고/노선 영역 분류기가 아니다. 한글 오독을 전부 제거하지 못하며 노란 인쇄 노선 등 정상 근거를 제외할 수도 있다. 같은 모델이 전체 번호를 반복 오독하면 시간 규칙만으로 막지 못한다. 점수는 보정된 정답 확률이 아니다.',
           '', '## 비슷한 번호 검사',f'실제 dataset1 7017 영상에 목표 7016 지정: 원시 일치 {rows[-1]["new_raw"]}, 판단 통과 {rows[-1]["new_match"]}.',
           '',f'71개 영상의 OCR를 목표만 바꾸어 {len(counter)}가지로 재생했다. 알려진 다른 노선, 한 자리 ±1 번호, 관측된 동일 길이 번호를 포함한다. 파일명 노선과 다른 목표에서 통과한 검토 대상은 **{len(totals["counterfactual_flags"])}건**이며 output의 counterfactual.csv에 있다. 화면 속 다른 버스도 있으므로 이 수치 자체는 오확정 건수나 오확정률이 아니다.',
           '', '## 속도',f'오프라인 {len(times)}프레임: 처리 p50 {totals["pipeline_ms"]["p50"]:.1f}ms, p95 {totals["pipeline_ms"]["p95"]:.1f}ms, 최대 {max(times):.1f}ms. 디코딩·그리기·인코딩은 제외.',
           '',f'기존과 같은 6개 영상을 원래 속도로 5fps 재생: 처리 {len(ages)}프레임, 폐기 {totals["realtime"]["dropped"]}프레임. 판단 시점 프레임 나이 p95 {totals["realtime"]["frame_age_p95_ms"]:.1f}ms, 최대 {max(ages):.1f}ms. 휴대폰 촬영·전송·음성 안내 지연을 포함한 시험은 아니다.',
           '', '## 검증', '단위·통합 테스트 26개 통과. 모든 결과 영상 디코딩 프레임 수, CSV 개수, crop 좌표 범위 및 저장 OCR 판정 재생 일치를 확인했다. 첫 판단 crop은 candidate_audit/에 보존했다.',
           '', '## 영상별 결과', '| 영상 | 이전 | 수정 후 | 첫 판단(초) |', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| [{r["video_path"].removeprefix("data/")}](../../{r["video_path"]}) | {"통과" if r["old_match"] else "보류"} | {"통과" if r["new_match"] else "보류"} | {r["first_match_s"] if r["first_match_s"] is not None else "—"} |')
    (WORK/'REPORT.md').write_text('\n'.join(lines)+'\n')
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>수정 전후 전체 비교</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px;line-height:1.6}article{border-top:2px solid #bbb;padding:20px 0}video{max-width:460px;width:100%;max-height:480px;background:#111}.crop{max-width:320px;min-height:60px;max-height:140px;object-fit:contain;image-rendering:pixelated}.context{max-height:450px;max-width:300px}code{overflow-wrap:anywhere}summary{cursor:pointer}input,select{font:inherit;padding:8px}</style>',
          f'<h1>수정 전후 비교: {totals["old_match"]}/70 → {totals["new_match"]}/70</h1><p>영상별 목표 번호 확인 성공률입니다. 독립 검증셋 정확도나 실제 탑승 안전도를 뜻하지 않습니다. <a href="REPORT.md">상세 보고서</a></p><p><input id="search" placeholder="파일명·노선 검색"><select id="filter"><option value="all">전체</option><option value="gain">새로 통과</option><option value="loss">통과→보류</option><option value="hold">수정 후 보류</option><option value="pass">수정 후 통과</option></select></p>']
    for r in rows:
        status='gain' if r['new_match'] and not r['old_match'] else 'loss' if r['old_match'] and not r['new_match'] else 'pass' if r['new_match'] else 'hold'
        page.append(f'<article data-status="{status}" data-pass="{int(r["new_match"])}"><h2>{r["folder"]} · {html.escape(r["target_route"])}</h2><code>{html.escape(r["video_path"].removeprefix("data/"))}</code><p>이전 {"통과" if r["old_match"] else "보류"} → 수정 후 {"통과" if r["new_match"] else "보류"} · 첫 판단 {r["first_match_s"]}초</p><details><summary>원본 / 처리 영상 재생</summary><video controls preload="none" src="../../{r["video_path"]}"></video><video controls preload="none" src="{base}{r["folder"]}/annotated.mp4"></video></details>')
        if r['new_match']:
            page.append(f'<p>첫 판단의 목표 crop (붉은 상자 위치)</p><a href="{base}candidate_audit/{r["folder"]}_context.jpg"><img loading="lazy" class="context" src="{base}candidate_audit/{r["folder"]}_context.jpg"></a><img loading="lazy" class="crop" src="{base}candidate_audit/{r["folder"]}_crop.png">')
        page.append('</article>')
    page.append('''<script>function filter(){const q=document.getElementById('search').value.toLowerCase(),v=document.getElementById('filter').value;for(const a of document.querySelectorAll('article'))a.hidden=!a.textContent.toLowerCase().includes(q)||!(v==='all'||v===a.dataset.status||v==='pass'&&a.dataset.pass==='1'||v==='hold'&&a.dataset.pass==='0');}for(const id of ['search','filter'])document.getElementById(id).addEventListener('input',filter);</script></html>''')
    (WORK/'review.html').write_text('\n'.join(page))
    (WORK/'HANDOFF.md').write_text('# 수정 후 평가 인계\n\n[REPORT.md](REPORT.md), [review.html](review.html) 및 output/target_20260929_run02/comparison_summary.json 참조.\n\n모델 교체 없이 crop 제외·차체 탐색 주기·최근 충돌 규칙 수정. 같은 dataset2 전체70개와 7017→7016 부정 사례1개 재추론. 이미 검토한 데이터이므로 독립 일반화 성능은 아니다. 신규 통과 crop과 다른 목표 재생의 모든 flag를 검토한 후 실제 오독 여부를 해석해야 한다.\n')
    print(json.dumps(totals,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
