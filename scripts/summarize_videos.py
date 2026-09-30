"""영상 배치 결과 검증·집계 및 원본/결과 비교 페이지 생성. 정답은 추론에 쓰지 않는다."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import imageio_ffmpeg
import numpy as np

from bus_route.detector import ROOT


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matching_summary(predictions, expected):
    matches = [p for p in predictions if expected and p['text'] == expected]
    times = sorted({float(p['timestamp_s']) for p in matches})
    per_track = defaultdict(set)
    for p in matches:
        if p['track_id'] not in ('', 'None', None):
            per_track[p['track_id']].add(float(p['timestamp_s']))
    return {
        'matching_sample_count': len(times),
        'matching_candidate_count': len(matches),
        'first_matching_candidate_s': times[0] if times else '',
        'last_matching_candidate_s': times[-1] if times else '',
        'matching_track_ids': json.dumps(sorted(per_track)),
        'max_matching_samples_same_track': max(map(len, per_track.values()), default=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workbench', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.workbench.exists():
        parser.error('기존 최종 결과를 덮어쓰지 않습니다.')
    inputs = read_csv(args.inputs)
    summaries = json.loads((args.results/'summary.json').read_text())
    settings = json.loads((args.results/'settings.json').read_text())
    inventory = json.loads(args.inventory.read_text())
    dataset = inventory.get('dataset', Path(inputs[0]['video_path']).parent.name)
    exclusion_text = ', '.join(Path(r['video_path']).name for r in inventory['excluded']) or '없음'
    diversity_text = (f"파일명 기준 {len({r['route_hint'] for r in inputs})}개 노선. 동일 노선 반복 파일은 서로 독립적인 사건이라고 가정하지 않는다.")
    if len(inputs) != len(summaries) or len(inputs) != inventory['included']:
        raise ValueError('입력 목록/완료 목록 개수가 다릅니다.')
    rows, latency, validation, observations = [], [], [], []
    status_counts = Counter()
    for i, (item, summary) in enumerate(zip(inputs, summaries), 1):
        folder = args.results/f'video_{i:02d}'
        if summary['video'] != item['video_path']:
            raise ValueError('입력 순서/파일 불일치')
        frames = read_csv(folder/'frames.csv')
        predictions = read_csv(folder/'predictions.csv')
        if len(frames) != summary['sample_count'] or len(predictions) != summary['prediction_count']:
            raise ValueError(f'CSV 개수 불일치: {folder}')
        # 생성된 MP4 전체 프레임을 디코딩해 정상 종료/프레임 개수를 확인한다.
        reader = imageio_ffmpeg.read_frames(str(folder/'annotated.mp4'))
        meta = next(reader)
        decoded = sum(1 for _ in reader)
        if decoded != len(frames):
            raise ValueError(f'결과 영상 프레임 수 불일치: {folder}')
        width, height = summary['metadata']['size']
        for p in predictions:
            x1, y1, x2, y2 = json.loads(p['text_box'])
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f'crop 좌표 범위 오류: {folder}')
        validation.append({'video_folder': folder.name, 'decoded_frames': decoded,
                           'prediction_rows': len(predictions), 'coordinate_bounds': 'pass'})
        stats = matching_summary(predictions, item['route_hint'])
        # 한글 접두 노선의 숫자 표기만 나온 경우를 별도 관찰. 전체 노선 정답 처리하지 않는다.
        suffix = re.sub(r'^[가-힣]+', '', item['route_hint'])
        partial = matching_summary(predictions, suffix) if suffix != item['route_hint'] else None
        row = {'video_folder': folder.name, **item,
               'sample_count': len(frames), 'candidate_count': len(predictions),
               'result': 'match_observed' if stats['matching_sample_count'] else 'no_match_observed',
               **stats, 'numeric_suffix_only_samples': partial['matching_sample_count'] if partial else 0,
               'first_numeric_suffix_s': partial['first_matching_candidate_s'] if partial else '',
               'route_format': 'korean_prefix' if partial is not None else 'ascii',
               'no_bus_samples': sum(f['status'] == 'no_bus' for f in frames),
               'no_text_candidate_samples': sum(f['status'] == 'no_text_candidate' for f in frames),
               'pipeline_p95_ms': summary['pipeline_ms']['p95'],
               'target_bus_verified': 'unreviewed', 'led_condition_verified': 'unreviewed'}
        rows.append(row)
        for p in predictions:
            kind = 'full_route_string' if p['text'] == item['route_hint'] else 'numeric_suffix_only' if partial and p['text'] == suffix else ''
            if kind:
                observations.append({'video_folder': folder.name, 'video_path': item['video_path'],
                                     'route_hint': item['route_hint'], 'match_kind': kind, **p})
        status_counts.update(f['status'] for f in frames)
        latency.extend(float(f['pipeline_ms']) for f in frames)
        print(f'검증 {i}/{len(inputs)}: {folder.name} frames={decoded}', flush=True)
    groups = {}
    for field in ['display_tag', 'view_tag', 'condition_tag', 'route_format']:
        groups[field] = {}
        for value in sorted({r[field] for r in rows}):
            group = [r for r in rows if r[field] == value]
            groups[field][value] = {'videos': len(group), 'match_observed': sum(r['matching_sample_count'] > 0 for r in group)}
    aggregate = {'inventory': inventory, 'analyzed_videos': len(rows), 'sampled_frames': len(latency),
                 'candidate_rows': sum(r['candidate_count'] for r in rows),
                 'videos_with_full_route_string': sum(r['matching_sample_count'] > 0 for r in rows),
                 'videos_without_full_route_string': sum(r['matching_sample_count'] == 0 for r in rows),
                 'videos_with_numeric_suffix_only': sum(r['numeric_suffix_only_samples'] > 0 and r['matching_sample_count'] == 0 for r in rows),
                 'videos_with_3_matching_samples_same_track': sum(r['max_matching_samples_same_track'] >= 3 for r in rows),
                 'groups': groups, 'frame_status_counts': dict(status_counts),
                 'pipeline_ms': {'p50': float(np.percentile(latency,50)), 'p95': float(np.percentile(latency,95)), 'max': max(latency)},
                 'interpretation': '파일명 전체 노선과 원시 후보 문자열 일치 관찰. 대상 버스/전면 LED/판독 가능 여부 미검수. 서비스 정확도 아님.'}
    write_csv(args.results/'video_summary.csv', rows)
    if observations:
        write_csv(args.results/'matching_observations.csv', observations)
    (args.results/'analysis_summary.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2))
    (args.results/'verification.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    shutil.copy2(args.inputs, args.results/'inputs.csv')
    shutil.copy2(args.inventory, args.results/'inventory.json')
    if inventory['excluded']:
        write_csv(args.results/'excluded.csv', inventory['excluded'])
    sources = [*sorted((ROOT/'core/bus_route').glob('*.py')), ROOT/'scripts/infer_video.py',
               ROOT/'scripts/summarize_videos.py', ROOT/'scripts/prepare_video_inputs.py', ROOT/'pyproject.toml']
    (args.results/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2))
    # 모든 결과 검증을 마친 뒤 최종 경로로 이동한다.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(args.results), args.output)
    args.workbench.mkdir(parents=True)
    run_id = args.output.name
    def result_link(row):
        return f'../../output/{run_id}/{row["video_folder"]}/annotated.mp4'
    def source_link(row):
        return '../../'+row['video_path']
    table = []
    for row in rows:
        outcome = '일치 있음' if row['matching_sample_count'] else '일치 없음'
        suffix_note = f' · 숫자 부분만 {row["numeric_suffix_only_samples"]}시점' if row['numeric_suffix_only_samples'] else ''
        first = str(row['first_matching_candidate_s'])+'초' if row['matching_sample_count'] else '—'
        table.append(f'| {row["video_folder"][6:]} | [{Path(row["video_path"]).name}]({source_link(row)}) | {outcome}{suffix_note} | {first} | {row["matching_sample_count"]} | [재생]({result_link(row)}) |')
    groups_md = '\n'.join(f'| {k} | {v["videos"]} | {v["match_observed"]} | {v["videos"]-v["match_observed"]} |' for k,v in groups['display_tag'].items())
    report = f'''# {dataset} 전체 영상 분석

분석 ID: `{run_id}`

## 결론

**{dataset}의 영상 {inventory['total']}개 중 {len(inventory['excluded'])}개를 제외한 {len(rows)}개 전체 영상을 분석했다.** 기존 모델·설정으로 초당 {settings['sample_fps']:g}프레임씩 실행했다. 추가 학습이나 검출 설정 조정은 하지 않았다. 파일명 전체 노선과 같은 원시 문자열이 나온 영상은 **{aggregate['videos_with_full_route_string']}개**, 한 번도 나오지 않은 영상은 **{aggregate['videos_without_full_route_string']}개**다.

이 수치는 **영상 안의 어느 후보에서든 같은 문자열이 나왔는지**를 센 것이다. 사람이 대상 버스의 번호인지 검수하지 않았으므로 인식 성공률로 확정하지 않는다. 실패 영상에 LED 획 누락·반사·가림·검출 실패가 얼마나 섞였는지 아직 분리하지 않았다. 따라서 이 결과만으로 LED 불량이나 모델 문제를 원인으로 단정하지 않는다.

[전체 원본·결과 비교 페이지](review.html)에서 일치 있음/없음, LED/인쇄, 파일명의 빛번짐/불량 조건으로 골라 확인할 수 있다. 첫 일치 시점으로 바로 이동할 수 있다. **번호가 보이는데 못 읽었는지, 원본에서도 획이 부족한지 확인하는 자료**로 사용한다.

## 범위와 기준

- 발견된 영상 {inventory['total']}개, 실행 {len(rows)}개, 제외 {len(inventory['excluded'])}개. 공항 표기로 제외한 파일: {exclusion_text}. 노선 번호나 색상만으로 제외하지 않았다.
- 단일 영상 중간 한 장이 아니라 각 영상 처음부터 끝까지 일정 간격으로 분석.
- {diversity_text} 차량 종류·촬영 조건의 다양성이 제한되어 있어 전체 버스로 일반화하지 않는다.
- 원본 전 프레임 추론은 아니다. 분석률 {settings['sample_fps']:g}fps, 총 {len(latency):,}표본, 문자 후보 {aggregate['candidate_rows']:,}개. 표본 사이의 LED 표시 순간은 놓칠 수 있다.
- `route_hint`는 파일명에서 추출한 노선. 기본 세 번째 항목이며 `세로/가로`가 있으면 다음 항목을 사용한다. 추론에는 전달하지 않고 사후 집계에만 사용. `data/annotations.csv`는 기존 소규모 crop 라벨이므로 이번 전체 영상 평가의 정답 목록으로 사용하지 않았다.
- 한글 접두 노선에서 `03`처럼 숫자만 나온 경우는 별도 표기하고 `서대문03` 전체 일치로 세지 않는다. `750A`, `750B`, `G6000` 등 접미/접두 문자도 원시 출력 그대로 비교한다.
- 같은 문자열이 동일 트랙에서 서로 다른 3개 이상 시점에 나온 영상은 {aggregate['videos_with_3_matching_samples_same_track']}개. 연속 일치나 정답 확정을 뜻하지 않는다.
- LED/PRINT/측면/빛번짐/불량 구분은 **파일명 태그**이며 이번 육안 판독 판정이 아니다. 태그가 없다고 영상 상태가 좋다는 뜻도 아니다.
- 여러 버스·광고·측면·인쇄 번호가 섞일 수 있다. 동일 버스 접근을 잘라 만든 파일의 중복도 미검증이므로 {len(rows)}개를 독립 사건 {len(rows)}건으로 세지 않는다.

| 파일명 표시 유형 | 실행 영상 | 전체 노선 일치 있음 | 일치 없음 |
|---|---:|---:|---:|
{groups_md}

## 검증과 속도

모든 입력 영상이 완료됐으며, 저장된 MP4 {len(rows)}개를 끝까지 다시 디코딩해 CSV와 프레임 수 일치, 모든 후보 좌표 범위를 확인했다. 검증 결과는 `verification.json`에 있다.

YOLO11n → 버스 상위 35% → CRAFT General → PARSeq, ByteTrack, GPU 환경은 기존 설정 유지. 모델 해시와 환경은 `settings.json`, `environment.txt`에 저장했다. 프레임 처리 p50 {aggregate['pipeline_ms']['p50']:.1f}ms, p95 {aggregate['pipeline_ms']['p95']:.1f}ms. 디코딩·표시·인코딩·전송은 제외한 값이며 실시간 동작의 검증 결과가 아니다.

## 원인 검토 순서

1. 원본에서 대상 번호를 사람이 읽을 수 있는 시점을 찾는다. 반사/획 누락/가림 등 관찰 내용과 시각을 기록한다.
2. 같은 시점의 결과 영상에서 버스 박스·상단 탐색 범위·문자 박스가 번호를 포함하는지 확인한다.
3. 번호가 충분히 담긴 crop에서도 틀린 경우에만 인식기 문제로 분류한다. LED 상태와 촬영 효과가 섞여 있을 때는 원인을 미확정으로 남긴다.

추가 수집·학습의 필요성과 주된 실패 원인은 이 원본 검토 전까지 보류한다. 전체 노선은 일치하지 않았지만 한글 접두 노선의 숫자 부분이 나온 영상은 {aggregate['videos_with_numeric_suffix_only']}개다. 한글 접두가 있는 영상과 없는 영상의 전체 문자열 일치율을 같은 난도로 해석하지 않는다.

## 영상별 목록

초록색 박스는 문자 후보이며 정답 표시가 아니다. 원본 링크가 브라우저에서 재생되지 않으면 로컬 플레이어로 열어 확인한다. 결과 MP4는 5fps/무음이다.

| 번호 | 원본 영상 | 전체 노선 문자열 관찰 | 첫 일치 | 일치 시점 수 | 결과 영상 |
|---|---|---|---:|---:|---|
{chr(10).join(table)}

## 저장 파일

- `inputs.csv`, `inventory.json`: 전체 실행 대상. 제외가 있을 때 `excluded.csv`에 근거 저장.
- `video_summary.csv`: 위 목록 및 파일명 조건별 정보. 대상 버스와 LED 상태는 `unreviewed`.
- `matching_observations.csv`: 일치 후보의 시각·트랙·박스·원시 문자열. 한글 접두 노선 숫자 부분 관찰도 별도 `match_kind`로 기록.
- 각 `video_XX/`: `annotated.mp4`, `frames.csv`, `predictions.csv`, `summary.json`.
- `analysis_summary.json`, `verification.json`, `source_sha256.json`: 집계·검증·실행 코드 해시.

기존 분석은 보존했고 이번 임시 산출물은 최종 검증 후 정리했다. 원본 영상은 변경하지 않았다.
'''
    (args.workbench/'REPORT.md').write_text(report)
    data = [{**r, 'name':Path(r['video_path']).name, 'source':source_link(r), 'annotated':result_link(r)} for r in rows]
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__DATASET__ 전체 영상 비교</title>
<style>body{font:15px system-ui;background:#f3f5f8;color:#1b2838;margin:24px}h1{font-size:25px}p{line-height:1.6}input,select,button{font:inherit;padding:8px;margin:4px;border:1px solid #bcc6d2;border-radius:6px}button{cursor:pointer}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}td:first-child{overflow-wrap:anywhere}#viewer{background:white;padding:18px;margin:20px 0;border-radius:10px}.players{display:flex;gap:18px;flex-wrap:wrap}.players>div{flex:1;min-width:270px}video{width:100%;max-height:560px;background:#111}.yes{color:#147442}.no{color:#a22}a{color:#1754ad}#wrap{overflow-x:auto}</style>
<h1>__DATASET__ 전체 영상 · 원본/자동 분석 비교</h1><p id="intro"></p><p>‘일치 있음’은 파일명 전체 노선과 같은 문자열이 나온 경우입니다. 대상 버스·전면 LED를 맞혔는지 아직 검수하지 않았습니다. 빛번짐/불량/측면/LED/PRINT는 파일명 태그입니다. 초록색 박스는 정답 표시가 아닙니다.</p><a href="REPORT.md">상세 보고서와 전체 목록</a>
<div id="viewer" hidden><h2 id="title"></h2><p id="detail"></p><button id="jump">첫 일치 시점으로 이동</button><div class="players"><div><h3>원본</h3><video id="original" controls preload="metadata"></video><a id="originalLink">원본 파일 열기</a></div><div><h3>자동 분석 (5fps, 무음)</h3><video id="result" controls preload="metadata"></video><a id="resultLink">결과 파일 열기</a></div></div><p>원본 MOV/HEVC가 브라우저에서 재생되지 않으면 로컬 플레이어로 여세요. 두 영상은 각각 재생하며, 버튼으로 동일 시간에 이동합니다.</p></div>
<div><input id="search" placeholder="노선·파일명 검색"><select id="outcome"><option value="all">전체 결과</option><option value="match_observed">일치 있음</option><option value="no_match_observed">일치 없음</option></select><select id="display"><option value="all">모든 표시 유형</option><option>LED</option><option>LED+PRINT</option><option>PRINT</option><option>unspecified</option></select><select id="condition"><option value="all">모든 촬영 조건</option><option value="빛번짐">파일명: 빛번짐</option><option value="빛반사">파일명: 빛반사</option><option value="불량">파일명: 불량</option><option value="side">파일명: 측면</option></select></div><p id="count"></p><div id="wrap"><table><thead><tr><th>원본 파일명</th><th>문자열 일치</th><th>첫 시점</th><th>일치 시점 수</th><th>영상</th></tr></thead><tbody id="body"></tbody></table></div>
<script>const rows=__DATA__; const stats=__STATS__;const $=id=>document.getElementById(id);let selected=null;
$('intro').textContent=`전체 ${stats.inventory.total}개 중 공항버스로 명시된 ${stats.inventory.excluded.length}개 제외, ${stats.analyzed_videos}개 분석. 일치 있음 ${stats.videos_with_full_route_string}개 / 없음 ${stats.videos_without_full_route_string}개. 초당 5장 분석.`;
function show(r){selected=r;$('viewer').hidden=false;$('title').textContent=r.name;$('detail').textContent=`기준 노선 ${r.route_hint} · ${r.display_tag} · 파일명 조건 ${r.condition_tag} · 숫자 부분만 관찰 ${r.numeric_suffix_only_samples}시점 · 대상 버스/LED 상태 미검수`;
$('original').src=r.source;$('result').src=r.annotated;$('originalLink').href=r.source;$('resultLink').href=r.annotated;$('jump').disabled=!r.matching_sample_count;$('viewer').scrollIntoView({behavior:'smooth'});}
$('jump').onclick=()=>{for(const id of ['original','result']){const v=$(id);if(v.readyState>=1)v.currentTime=Number(selected.first_matching_candidate_s);}};
function render(){const list=rows.filter(r=>(r.name.toLowerCase().includes($('search').value.toLowerCase()))&&($('outcome').value==='all'||r.result===$('outcome').value)&&($('display').value==='all'||r.display_tag===$('display').value)&&($('condition').value==='all'||($('condition').value==='side'?r.view_tag==='side':r.condition_tag.includes($('condition').value))));$('count').textContent=`${list.length}개 표시`;$('body').replaceChildren();for(const r of list){const tr=document.createElement('tr');for(const [text,cls] of [[r.name,''],[r.matching_sample_count?'일치 있음':'일치 없음',r.matching_sample_count?'yes':'no'],[r.matching_sample_count?r.first_matching_candidate_s+'초':'—',''],[String(r.matching_sample_count)+(r.numeric_suffix_only_samples?' (숫자 부분 '+r.numeric_suffix_only_samples+')':''),'']]){const td=document.createElement('td');td.textContent=text;td.className=cls;tr.append(td);}const td=document.createElement('td'),b=document.createElement('button');b.textContent='비교 보기';b.onclick=()=>show(r);td.append(b);tr.append(td);$('body').append(tr);}}
for(const id of ['search','outcome','display','condition'])$(id).addEventListener('input',render);render();</script></html>'''
    page = page.replace('__DATASET__', html.escape(dataset))
    page = page.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')).replace('__STATS__',json.dumps(aggregate,ensure_ascii=False).replace('<','\\u003c'))
    (args.workbench/'review.html').write_text(page)
    (args.workbench/'HANDOFF.md').write_text(f'''# 전체 영상 실행 인계

[REPORT.md](REPORT.md), [review.html](review.html) 참고.

- {dataset} {inventory['total']}개 중 공항 표기 {len(inventory['excluded'])}개를 제외한 {len(rows)}개 전체를 {settings['sample_fps']:g}fps로 실행.
- {diversity_text} 차량 종류의 다양성 제한을 함께 해석.
- 전체 노선 문자열 일치 관찰 {aggregate['videos_with_full_route_string']}개, 미관찰 {aggregate['videos_without_full_route_string']}개. 대상 버스·LED 판독 가능 여부는 미검수.
- LED·인쇄·측면·빛번짐·불량은 파일명 태그 기준. 원인 판정하지 않음.
- 한글 접두 노선의 숫자 부분 일치는 별도로 기록하며 전체 노선 정답으로 치환하지 않음.
- 다음은 사용자가 원본/결과를 비교하여 영상 정보 부족과 검출·인식 실패를 구분하는 단계. 추가 학습이나 수집을 자동 결정하지 않음.
- 현재 실행 모델/설정은 이전과 동일. 전체 결과는 `output/{run_id}/`. 원본/이전 결과 보존.
''')
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
