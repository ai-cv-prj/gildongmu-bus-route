"""목표 노선 실험의 원시 일치/보수 판단/지연을 분리해 최종 보고한다."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import imageio_ffmpeg
import numpy as np
from bus_route.detector import ROOT
from bus_route.target import TargetMatcher,exact_route


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def replay_decisions(folder,target):
    # 같은 OCR 결과를 다른 목표로 평가. 검출/인식은 목표와 독립.
    frames=read_csv(folder/'frames.csv');preds=read_csv(folder/'predictions.csv');by_frame={}
    for p in preds:
        p={**p,'token_scores':json.loads(p['token_scores']),'eligible':p['eligible']=='True'}
        by_frame.setdefault(p['sample_index'],[]).append(p)
    matcher=TargetMatcher(target,['7016','7017']);matched=[]
    for f in frames:
        for j,bus in enumerate(json.loads(f['buses'])):
            observations=[p for p in by_frame.get(f['sample_index'],[]) if int(p['bus_index'])==j]
            result=matcher.update(bus['track_id'],float(f['timestamp_s']),observations,bus['bus_box'],not bus['candidate_truncated'])
            if result['state']=='matched_candidate':matched.append({'timestamp_s':f['timestamp_s'],'track_id':bus['track_id']})
    return matched


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--staging',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workbench',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or args.workbench.exists():p.error('기존 산출물 덮어쓰기 금지')
    stage=args.staging;results=stage/'results';inputs=read_csv(stage/'inputs.csv');sums=json.loads((results/'summary.json').read_text())
    validation=[];times=[];rows=[];counterfactual=[]
    old=read_csv(ROOT/'output/dataset2_20260929_run01/video_summary.csv')
    recount=read_csv(stage/'recount/recount.csv');accepted_old={r['video_path']:r['accepted_match']=='True' for r in recount}
    for i,(inp,s) in enumerate(zip(inputs,sums),1):
        folder=results/f'video_{i:02d}';frames=read_csv(folder/'frames.csv');preds=read_csv(folder/'predictions.csv')
        reader=imageio_ffmpeg.read_frames(str(folder/'annotated.mp4'));next(reader);count=sum(1 for _ in reader)
        assert count==len(frames)==s['sample_count'];assert len(preds)==s['prediction_count']
        w,h=s['metadata']['size']
        for pred in preds:
            a,b,c,d=json.loads(pred['text_box']);assert 0<=a<c<=w and 0<=b<d<=h
        validation.append({'folder':folder.name,'frames':count,'predictions':len(preds),'coordinates':'pass'})
        times.extend(float(f['pipeline_ms']) for f in frames)
        row={**inp,'folder':folder.name,'raw_match':s['target_match_samples']>0,'raw_match_samples':s['target_match_samples'],
             'first_raw_s':s['first_exact_s'],'first_rule_match_s':s['first_matched_candidate_s'],
             'rule_match':s['first_matched_candidate_s'] is not None,'body_match_samples':s['body_match_samples'],
             'conflict_seen':s['decision_counts'].get('conflict',0)>0,'pipeline_p95_ms':s['pipeline_ms']['p95'],
             'baseline_accepted_match':accepted_old.get(inp['video_path'],False) if inp['test_kind']=='filename_target_simulation' else None}
        rows.append(row)
        for target in ['7016','7017']:
            matched=replay_decisions(folder,target)
            counterfactual.append({'video_path':inp['video_path'],'filename_route':inp['route_hint'],'target_route':target,
                                   'matched_candidate_samples':len(matched),'first_s':matched[0]['timestamp_s'] if matched else '',
                                   'filename_same_target':exact_route(inp['route_hint'],target)})
    assert len(rows)==len(inputs)==71
    positives=[r for r in rows if r['test_kind']=='filename_target_simulation'];negative=rows[-1]
    replay=json.loads((stage/'realtime/summary.json').read_text());ages=[];replay_times=[];replay_valid=[]
    for i,s in enumerate(replay,1):
        folder=stage/'realtime'/f'video_{i:02d}';frames=read_csv(folder/'frames.csv')
        reader=imageio_ffmpeg.read_frames(str(folder/'annotated.mp4'));next(reader);n=sum(1 for _ in reader)
        assert n==s['rendered_frame_count'] and n-len(frames)==s['dropped_samples']
        ages.extend(float(f['frame_age_ms']) for f in frames);replay_times.extend(float(f['pipeline_ms']) for f in frames)
        replay_valid.append({'video':s['video'],'processed':len(frames),'rendered':n,'dropped':s['dropped_samples']})
    totals={'videos':len(rows),'dataset2_videos':len(positives),'frames':sum(s['sample_count'] for s in sums),
            'raw_match_dataset2':sum(r['raw_match'] for r in positives),'rule_match_dataset2':sum(r['rule_match'] for r in positives),
            'body_match_videos':sum(r['body_match_samples']>0 for r in positives),
            'gained_raw_videos':[r['video_path'] for r in positives if r['raw_match'] and not r['baseline_accepted_match']],
            'lost_raw_videos':[r['video_path'] for r in positives if not r['raw_match'] and r['baseline_accepted_match']],
            'negative_7017_target_7016':negative,
            'pipeline_ms':{'p50':float(np.percentile(times,50)),'p95':float(np.percentile(times,95)),'max':max(times)},
            'replay':{'videos':len(replay),'processed_frames':len(ages),'dropped_samples':sum(s['dropped_samples'] for s in replay),
                      'frame_age_p95_ms':float(np.percentile(ages,95)),'pipeline_p95_ms':float(np.percentile(replay_times,95)),
                      'max_frame_age_ms':max(ages)},
            'counterfactual_flags':[r for r in counterfactual if r['matched_candidate_samples'] and not r['filename_same_target']]}
    write_csv(results/'target_video_summary.csv',rows);write_csv(results/'counterfactual_7016_7017.csv',counterfactual)
    dump(results/'target_summary.json',totals);dump(results/'verification.json',{'offline':validation,'replay':replay_valid})
    for name in ['inputs.csv','inventory.json','replay_inputs.csv']:shutil.copy2(stage/name,results/name)
    shutil.move(stage/'recount',results/'recount');shutil.move(stage/'realtime',results/'realtime')
    sources=[*sorted((ROOT/'core/bus_route').glob('*.py')),ROOT/'scripts/infer_target_video.py',ROOT/'scripts/recount_targets.py',ROOT/'scripts/report_target_video.py']
    dump(results/'source_sha256.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True);shutil.move(results,args.output);args.workbench.mkdir(parents=True)
    base='../../output/'+args.output.name+'/'
    links=[]
    for r in rows:
        name=Path(r['video_path']).name
        links.append(f"| {r['folder']} | [{name}](../../{r['video_path']}) | {r['target_route']} | {'있음' if r['raw_match'] else '없음'} | {r['first_rule_match_s'] if r['rule_match'] else '보류'} | {r['body_match_samples']} | [재생]({base}{r['folder']}/annotated.mp4) |")
    report=f'''# 목표 버스 식별 — 지역 접두 생략·차체 탐색·보수 판단

## 결론

1~3 구현 완료. **부분 번호를 목표 번호로 채우지 않는다.** 동일 트랙의 온전한 전체 번호가 서로 다른 3개 시점(최소 0.4초, 최근 1.2초)에 나오고, 현재 프레임에도 신뢰 기준을 만족하는 전체 일치가 있어야 `matched_candidate`가 된다. 이름 그대로 검증용 일치 후보이며 실제 탑승 확인을 보장하는 출력이 아니다.

실제 영상: dataset2 전체 70개 + dataset1의 7017 영상에 목표 7016 지정 1개. dataset2 원시 번호 일치 **{totals['raw_match_dataset2']}/70**, 보수 판단 조건 통과 영상 **{totals['rule_match_dataset2']}/70**. 이 둘을 같은 성공률로 비교하지 않는다. 검수되지 않은 다른 버스·광고 후보가 포함될 수 있고, 충돌 시에는 맞는 버스도 보류할 수 있다.

7017 영상에서 목표 7016의 원시 일치: **{negative['raw_match']}**, 판단 조건 통과: **{negative['rule_match']}**. 이 한 영상만으로 오탑승 방지를 보장하지 않는다. 71개 영상 각각에 목표 7016/7017을 바꾸어 적용한 저장 예측 재생 결과는 `counterfactual_7016_7017.csv`에 있다. 파일명과 다른 목표의 판단 후보가 나온 검토 대상은 **{len(totals['counterfactual_flags'])}건**이다. 파일명과 다른 버스가 실제 화면에 있을 수 있으므로 이 값은 오확정률이 아니다.

[전체 비교 페이지](review.html)

## 1. 번호 일치 기준

`마포07`과 `07`은 일치, `7`은 불일치. `110A/110B`, `M6628/6628`은 서로 다르다. 명시적 다른 지역 접두를 일괄 삭제하지 않는다. API 후보가 같은 표시 번호로 겹치면 입력 단계에서 거절한다. 기존 원시 예측을 재추론 없이 재집계한 결과: **dataset1 48→52/79, dataset2 44→45/70**. 학습 성능 개선이 아니라 사용자 승인 기준 변경이다. 원래 CSV/보고서는 유지하고 `recount/`에 새 계산을 저장했다.

## 2. 상단·차체 탐색

상단 35%는 매 분석 프레임, 차체 y=25~100%는 트랙별 0.6초마다 탐색한다. 따라서 측면에 적힌 인쇄 번호도 탐색 범위에 들어간다. 차체 CRAFT canvas 960, 후보 OCR은 최대 버스당 24개/배치 16개로 제한한다. 누락된 후보가 있으면 그 프레임은 판단 조건을 통과하지 못한다.

차체 영역에서 목표 문자열이 관찰된 영상은 **{totals['body_match_videos']}개**. 영역만으로 전면/측면 또는 LED/PRINT를 확정하지 않는다. 기록의 `upper/body`는 탐색 좌표 구분이다. 원본 번호·기울기·차량 대응을 확인할 수 있도록 모든 후보와 원시 문자열을 저장했다. 시야의 다른 버스 박스에도 포함되는 문자 후보는 판단 근거에서 제외한다.

기존 수정 기준 45/70 대비 새롭게 일치가 나온 영상 {len(totals['gained_raw_videos'])}개, 사라진 영상 {len(totals['lost_raw_videos'])}개. 차체 추가 외 OCR 배치화·후보 제한도 달라졌으므로 변화를 전부 측면 탐색 효과로 단정하지 않는다.

## 3. 비슷한 번호를 위한 보류 규칙

- `701`, `701?`, `?016`을 합쳐 7016으로 확정하지 않는다. 부분 근거 상태로만 기록한다.
- 동일 트랙에서 다른 전체 번호(예: 7017)가 충분한 점수로 한 번이라도 나오면 영상 끝까지 충돌 상태를 유지한다. 시간 경과나 목표와의 유사도로 충돌을 지우지 않는다.
- 다른 트랙의 결과를 합치지 않는다. 추적 ID가 없거나 위치가 크게 뛰거나 관측 간격이 벌어지면 누적 근거를 사용하지 않는다.
- 전체 일치 기준은 각 문자 및 EOS의 최소 점수 0.9. 충돌 기록은 0.75 이상. **이 점수는 검증된 정답 확률이 아니며 임계값의 현장 최적성은 미검증이다.**
- 전체 문자열을 모델이 반복해서 잘못 읽고 충돌 단서도 안 나오면 이 규칙만으로 걸러낼 수 없다. 번호 전용 검출·독립 검수·더 많은 혼동 사례로 검증이 필요하다. 음성·탑승 명령은 구현하지 않았다.
- 광고/차량 관리 번호가 목표와 같은 길이이면 보수적으로 충돌 처리할 수 있다. 잘못된 확정을 줄이는 대신 보류가 늘어나는 초기 정책이다.

## 속도와 실제 재생 시험

오프라인 {totals['frames']}프레임: 처리 p50 {totals['pipeline_ms']['p50']:.1f}ms / p95 {totals['pipeline_ms']['p95']:.1f}ms / 최대 {totals['pipeline_ms']['max']:.1f}ms. 처리 시간은 검출·CRAFT·OCR·판단 포함, 디코딩·표시·인코딩 제외.

별도 6개 영상은 **원래 재생 속도, 5fps 입력, 최신 1장 대기열**로 시험했다. 처리 {totals['replay']['processed_frames']}표본, 폐기 {totals['replay']['dropped_samples']}표본. 결과 시점의 프레임 나이 p95 **{totals['replay']['frame_age_p95_ms']:.1f}ms**, 최대 **{totals['replay']['max_frame_age_ms']:.1f}ms**. 처리 p95 **{totals['replay']['pipeline_p95_ms']:.1f}ms**. 처리 후 나이가 300ms를 넘으면 후보 판단은 보류한다. 과부하 때 오래된 프레임을 쌓지 않고 버리며, 영상의 해당 시점은 `NOT INFERRED`로 표시한다.

이는 노트북 저장 영상 재생 시험이다. 스마트폰 촬영·전송·TTS까지 포함한 실시간 동작 검증은 아니다. 5fps를 초과한 목표 성능도 입증하지 않았다. 재생 테스트 영상/CSV는 `realtime/`에 보존했다.

## 현재 모델이 최선인가

**비교 실험을 하지 않았으므로 최선이라고 말할 수 없다.** 공식 자료에서 확인한 대안은 다음과 같다. 이 실험에서 대체 모델을 실제 실행하지는 않았다.

| 후보 | 비교 목적 | 한계 |
|---|---|---|
| 번호 영역 전용 YOLO + 현재 PARSeq | 차체 광고를 줄이고 번호 crop만 OCR에 전달. 탐색 비용과 충돌 오탐 줄이기 | 번호 영역 라벨과 별도 학습 필요. 이번 자료에서 우수성 미측정 |
| PP-OCRv5 검출/인식, mobile/server 구성 | CRAFT 누락과 속도·인식 조합 비교 | 공식 일반 OCR 벤치마크가 버스 LED 성능을 보장하지 않음 |
| PARSeq-tiny / CRNN | 같은 정확한 crop에서 속도·혼동 번호 오류 비교 | 속도 향상/정확도 개선은 로컬 실측 전까지 미확인 |

공식 근거: [Ultralytics 검출 학습](https://docs.ultralytics.com/tasks/detect/), [PP-OCRv5 검출 구성](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html), [PP-OCRv5 인식 구성](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html), [PARSeq 공식 모델 목록](https://github.com/baudm/parseq#pretrained-models-via-torch-hub).

현재 구조에서는 번호 영역 전용 검출기를 다음 비교 대상으로 우선 검토할 만하다. 이는 넓어진 차체 영역의 불필요한 문자 후보를 줄일 수 있다는 설계 판단이며 성능 향상 입증은 아니다. 목표 번호를 보고 손상 숫자를 생성하는 방식보다 실제로 보이는 전면/측면 전체 번호의 근거를 확보하는 쪽을 유지한다.

## 재실행

```bash
.venv/bin/python scripts/infer_target_video.py --video data/video/dataset1/20260915_갤럭시S24+_7017_LED.mp4 --target-route 7016 --competitors 7017 --output output/target_manual_next --device cuda --realtime
.venv/bin/python -m unittest discover -s scripts/tests -v
```

CSV 실행은 `--videos-csv ... --target-from-manifest`: `target_route` 열로 사용자가 입력한 목표를 모의한다. 파일명 라벨은 모델에 전달하지 않으며 결과를 정답으로 치환하지 않는다. 사용 모듈은 `core/bus_route/target.py`, 검출기/인식기/영상 처리기와 `scripts/infer_target_video.py`다. 기존 기본 실행 스크립트는 유지했다.

## 영상별 확인

| 폴더 | 원본 | 지정 목표 | 원시 번호 일치 | 첫 판단 후보(초) | 차체 일치 시점 수 | 결과 |
|---|---|---|---|---:|---:|---|
{chr(10).join(links)}
'''
    (args.workbench/'REPORT.md').write_text(report)
    data=[{**r,'name':Path(r['video_path']).name,'source':'../../'+r['video_path'],'result_video':base+r['folder']+'/annotated.mp4'} for r in rows]
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>목표 버스 판단 비교</title><style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;background:#f4f6f8}p{line-height:1.6}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}button,input,select{padding:8px;font:inherit;margin:5px}#players{display:flex;gap:20px;flex-wrap:wrap}video{width:100%;max-height:550px;background:#111}#players>div{flex:1;min-width:300px}td:first-child{overflow-wrap:anywhere}</style><h1>목표 버스 · 상단/차체 탐색과 보수 판단</h1><p>부분 번호를 채우지 않습니다. ‘판단 후보’도 실제 탑승 확정이 아닙니다. 주황색은 상단 탐색, 자주색은 차체 탐색, 초록색은 문자 후보입니다. 차체 영역은 측면 인쇄 번호도 포함하지만 위치만으로 전면/측면을 확정하지 않습니다.</p><a href="REPORT.md">실행 수치·보류 규칙·모델 대안 보고서</a><section id="viewer" hidden><h2 id="title"></h2><p id="detail"></p><button id="jump">첫 일치 시점으로 이동</button><div id="players"><div><h3>원본</h3><video id="original" controls></video><a id="sourceLink">원본 파일</a></div><div><h3>처리 결과</h3><video id="processed" controls></video><a id="resultLink">결과 파일</a></div></div></section><p><input id="search" placeholder="노선·파일명"><select id="filter"><option value="all">전체</option><option value="rule">판단 후보 있음</option><option value="raw">원시 일치만 있음</option><option value="none">일치 없음</option><option value="body">차체 일치 있음</option><option value="conflict">충돌 기록 있음</option><option value="negative">7017→목표7016 검사</option></select></p><p id="count"></p><table><thead><tr><th>영상</th><th>목표</th><th>원시 일치</th><th>판단 후보</th><th>차체 일치</th><th>재생</th></tr></thead><tbody id="rows"></tbody></table><script>const rows=__DATA__;const $=id=>document.getElementById(id);let selected;function show(r){selected=r;$('viewer').hidden=false;$('title').textContent=r.name;$('detail').textContent=`목표 ${r.target_route} / 파일명 노선 ${r.route_hint} · 원시 일치 ${r.raw_match_samples}시점 · 충돌 기록 ${r.conflict_seen}`;$('original').src=r.source;$('processed').src=r.result_video;$('sourceLink').href=r.source;$('resultLink').href=r.result_video;$('jump').disabled=r.first_raw_s===null;$('viewer').scrollIntoView({behavior:'smooth'});} $('jump').onclick=()=>{for(const id of ['original','processed'])if($(id).readyState>=1)$(id).currentTime=selected.first_raw_s;};function render(){const v=$('filter').value;const list=rows.filter(r=>r.name.toLowerCase().includes($('search').value.toLowerCase())&&(v==='all'||v==='rule'&&r.rule_match||v==='raw'&&r.raw_match&&!r.rule_match||v==='none'&&!r.raw_match||v==='body'&&r.body_match_samples>0||v==='conflict'&&r.conflict_seen||v==='negative'&&r.test_kind!=='filename_target_simulation'));$('count').textContent=list.length+'개';$('rows').replaceChildren();for(const r of list){const tr=document.createElement('tr');for(const text of [r.name,r.target_route,r.raw_match?'있음':'없음',r.rule_match?r.first_rule_match_s+'초':'보류',r.body_match_samples]){const td=document.createElement('td');td.textContent=text;tr.append(td);}const td=document.createElement('td'),b=document.createElement('button');b.textContent='비교';b.onclick=()=>show(r);td.append(b);tr.append(td);$('rows').append(tr);}}for(const id of ['search','filter'])$(id).addEventListener('input',render);render();</script></html>'''
    (args.workbench/'review.html').write_text(page.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
    (args.workbench/'HANDOFF.md').write_text(f'''# 목표 버스 판단 인계

[REPORT.md](REPORT.md), [review.html](review.html) 참고.

- 지역 접두 생략 기준 재집계: dataset1 52/79, dataset2 45/70. 원래 예측 유지.
- 상단+차체 탐색, 목표 번호 입력, 부분 판독 보류, 전체 번호 충돌 영구 보류, 최신 1프레임 재생 추가.
- dataset2 원시 일치 {totals['raw_match_dataset2']}/70, 실험용 보수 조건 통과 {totals['rule_match_dataset2']}/70. 7017 영상 목표7016 통과: {negative['rule_match']}.
- 실시간 재생6개: p95 프레임 나이 {totals['replay']['frame_age_p95_ms']:.1f}ms, 5fps 입력. 휴대폰 전송 시험 아님.
- 원시 전체 문자열도 반복 오독될 수 있어 실제 오탑승 방지 보장 불가. 번호 전용 검출 등 후보 정제와 혼동 사례 검수가 필요.
- 차체 영역을 측면/PRINT로 자동 라벨링하지 않음. 입력 태그·원본을 검수해야 한다.
- 다음 모델 후보·공식 출처는 보고서 참조. 대체 모델을 이번 실행에서 시험하지 않음.
''')
    print(json.dumps(totals,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
