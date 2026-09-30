"""Publish the completed evaluation at the existing workbench entry point."""
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path
import re

from evaluate import ROOT, OUT, BASE, read, write

WB=ROOT/'workbench/plate_filter_20260929_run01'


def main():
    offline=json.loads((OUT/'offline_evaluation.json').read_text())
    realtime=json.loads((OUT/'realtime_evaluation.json').read_text())
    evidence=json.loads((OUT/'evidence/review.json').read_text())
    assert all(r['verdict']!='needs_review' for r in evidence), 'Review new evidence before publishing'
    WB.mkdir(parents=True,exist_ok=True)
    rt=realtime['timing'];off=offline['timing']
    reason_name={'likely_registration_plate':'등록번호판 의심','embedded_phone_number':'전화번호 일부',
                 'search_boundary_clipped':'탐색 경계 잘림','small_clipped_or_ambiguous_bus':'크기·잘림·버스 귀속 모호',
                 'implausible_character_density':'문자 폭 불일치','':'허용'}
    selected=lambda s:[x for x in s['reviewed_diagnostics'] if x['verdict']=='false_recognition']
    blocked=lambda s:sum(not x['recognized'] for x in selected(s))
    def delta(s,key):
        return ', '.join(f"{r['folder']} {r['target']}" for r in s[key]) or '없음'
    metrics=[['평가','기존','변경 후','증가','감소'],
             ['전체 재추론',str(offline['old_recognized']),str(offline['new_recognized']),str(len(offline['gains'])),str(len(offline['losses']))],
             ['원래 속도 재생',str(realtime['old_recognized']),str(realtime['new_recognized']),str(len(realtime['gains'])),str(len(realtime['losses']))]]
    md=['# 번호판·광고 전화번호 제외 검증 · 2026-09-29','',
        f"활성 143개 전체를 원본에서 재추론했다. offline **{offline['new_recognized']}/143 ({offline['new_recognized']/143:.1%})**, 원래 속도 replay **{realtime['new_recognized']}/143 ({realtime['new_recognized']/143:.1%})**. 비교 기준은 단일 관측을 포함한 기존 105/143이다.",'',
        '| 평가 | 기존 | 변경 후 | 증가 | 감소 |','|---|---:|---:|---:|---:|']
    md += ['| '+' | '.join(r)+' |' for r in metrics[1:]]
    md += ['',f"offline 증가: {delta(offline,'gains')}. 감소: {delta(offline,'losses')}.",
           f"realtime 증가: {delta(realtime,'gains')}. 감소: {delta(realtime,'losses')}.",'',
           '## 적용한 변경','',
           '- 기존의 상대 높이 ≥0.72 / 채도 S≥100 / 노란색 비율 ≥0.45 규칙은 유지한다. 확장 규칙은 상대 높이 ≥0.55, HSV H=12..40 / S≥25 / V≥30인 노란 바탕 비율 ≥0.45에 더해 Otsu 밝기 분할에서 crop 가장자리의 밝은 픽셀 비율 ≥0.6을 요구한다. 위치만으로 차체 하단을 삭제하지 않는다.',
           '- 첫 개발 실행에서 색상만 확장했을 때 정상 측면 PRINT 03·LED 6713의 4개 관측이 차단되는 회귀를 발견했다. 해당 실행의 성능 측정을 중단하고 밝은 바탕 조건을 추가했다. 4개 실제 crop을 회귀 fixture에 보존하고 최종 코드로 전체 offline/realtime을 다시 실행했다. 초기 106/143 결과는 최종 성능에 사용하지 않는다.',
           '- 적격 4자리 숫자 중 문자 높이/버스 높이 ≤0.065, 상대 높이 ≥0.25, 원시 점수 ≥0.75인 작은 차체 표기는 좌우 두 문맥을 확인한다. 문맥 범위는 글자 높이의 1배/4배를 좌우로 확장하며 버스 검출 박스 밖도 확인한다.',
           '- 문맥에서 4자리-4자리 형식이 읽히고 원시 4자리와 하이픈 점수 ≥0.9이면 전화번호 일부로 제외한다. 문맥의 나머지 숫자 전체를 확정할 필요는 없으며 문맥 OCR을 노선 근거에 추가하지 않는다. 원시 OCR·문맥 OCR·거절 이유는 별도 필드로 남긴다.',
           '- 목표/파일명/영상별 좌표/금지 번호 목록을 필터 입력으로 쓰지 않는다. 단일 관측 ≥0.98, 반복, 충돌, 트랙, 최신성 300ms 조건은 유지했다. 추가 모델·새 데이터는 사용하지 않았다.','',
           '## 세 숫자의 실제 출처와 차단','',
           '**이전 설명 정정:** 1577은 뒤쪽 7611 버스 광고 전화번호 1577-1987의 앞부분이며, 9350은 같은 뒤쪽 버스의 그림자 속 등록번호판이다. 기존 전면 번호판/별도 차체 식별번호 설명은 실제 OCR 좌표와 맞지 않았다. 주황색 버스 박스와 빨간 OCR 박스를 원본 프레임에 그려 확인했다.','',
           '| 모드 | 숫자 | 원시 OCR 관측 | 적격 관측 | 해당 숫자를 목표로 인식 | 거절 이유 |','|---|---|---:|---:|---|---|']
    for s in [offline,realtime]:
        for r in s['reviewed_diagnostics']:
            if r['id'] in [19,24,25]:
                md.append(f"| {s['mode']} | {r['target']} | {r['raw_ocr_samples']} | {r['eligible_samples']} | {'예' if r['recognized'] else '아니오'} | {r['rejection_reasons']} |")
    md += ['', '세 숫자는 원시 OCR 로그에 남아도 노선 판정 근거에서는 제외되는지 별도로 검사했다.', '',
           '## 정상 측면 및 유사 번호 보호','',
           '| 모드 | 영상 | 목표 | 기존 근거 시각 | 원시 관측 | 적격 관측 | 영상 인식 |','|---|---|---|---:|---:|---:|---|']
    for s in [offline,realtime]:
        for r in s['protected_anchors']:
            if r['id'] in [0,2,4,5,7,8,9]:
                md.append(f"| {s['mode']} | {r['folder']} | {r['target']} | {r['anchor_s']} | {r['anchor_raw']} | {r['anchor_eligible']} | {'유지' if r['recognized'] else '보류'} |")
    md += ['',f"전체 프레임의 기존 적격 목표 관측 중 새 필터가 제외한 관측: offline {len(offline['newly_filtered_target_observations'])}개, realtime {len(realtime['newly_filtered_target_observations'])}개. 단순 영상별 인식 유지뿐 아니라 실제 관측의 회귀도 검사했다.",
           '',f"실제 7017 영상에 목표 7016을 넣은 별도 반례: offline={offline['negative_control']['new_recognized']}, realtime={realtime['negative_control']['new_recognized']}. 분모 143에는 포함하지 않았다. 부분 701을 7016으로 채우지 않는 단위 검사와 문맥 거절 전용 통합 검사도 통과했다.",'',
           '## 실시간 지연','',
           'NVIDIA RTX 5080 Laptop GPU, torch 2.7.1+cu128, 영상 전체 5fps. offline과 realtime을 순서대로 실행했고 같은 GPU에 다른 추론을 병행하지 않았다. 처리 시간은 검출 시작부터 OCR·판정 종료까지이며 프레임 나이는 그 종료 시각에서 원래 재생 시각을 뺀 값이다. 처리 영상 렌더·인코딩 종료 시각이나 모델 준비 시간은 해당 지표에 포함하지 않는다. 휴대폰 추론·네트워크·음성 안내는 측정 범위 밖이다.','',
           '| 지표 | 이번 실제 원래 속도 재생 |','|---|---:|',
           f"| 처리 프레임 | {realtime['processed_frames']} |",
           f"| 처리 p50 / p95 / 최대 ms | {rt['pipeline_ms']['p50']:.2f} / {rt['pipeline_ms']['p95']:.2f} / {rt['pipeline_ms']['max']:.2f} |",
           f"| 프레임 나이 p50 / p95 / 최대 ms | {rt['frame_age_ms']['p50']:.2f} / {rt['frame_age_ms']['p95']:.2f} / {rt['frame_age_ms']['max']:.2f} |",
           f"| 300ms 초과 프레임 | {realtime['frame_age_over_300ms_samples']} |",
           f"| 샘플 폐기 | {realtime['dropped_samples']} |",
           f"| incomplete 버스 판정 | {realtime['incomplete_bus_decisions']} |",
           f"| 추가 문맥 OCR crop 수 | {realtime['context_ocr_calls']} |",'',
           '판정 직전의 complete/300ms 검사를 우회한 인식은 0건이다. 판정 종료 후 기록한 인식 프레임 나이 최대는 300.008ms로, 검사와 기록 시각 사이의 처리 시간을 포함한다.', '',
           '기존 replay는 프레임 나이 p95 161.14ms / 최대 411.29ms, 폐기 1개였다. 재실행의 지연 차이는 추가 OCR과 시스템 변동을 함께 포함하며, 같은 조건의 반복 측정으로 분리 추정한 값은 아니다. 모든 장면이 200ms 안에 처리된다는 뜻이 아니며 다음 느린 영상도 포함했다.','',
           '| 영상 | 처리 p50 / p95 / 최대 ms | 나이 p95 / 최대 ms | 폐기 |','|---|---|---|---:|']
    for r in realtime['slowest_videos']:
        p,a=r['pipeline_ms'],r['frame_age_ms']
        md.append(f"| {r['folder']} · {Path(r['video']).name} | {p['p50']:.1f} / {p['p95']:.1f} / {p['max']:.1f} | {a['p95']:.1f} / {a['max']:.1f} | {r['dropped_samples']} |")
    md += ['', '## 남은 오류와 검사 범위','',
           f"기존에 육안 검토한 오류 12조합 중 이번 offline {blocked(offline)}조합, realtime {blocked(realtime)}조합이 차단됐다. 두 조합(11/마포11)은 같은 물리적 오독이므로 조합 수를 독립 오류 수로 해석하지 않는다. 나머지 오류는 아래 근거에 그대로 공개한다.",'',
           '| 영상 | 시험 목표 | 이전 육안 분류 | offline 인식 | realtime 인식 |','|---|---|---|---|---|']
    for a,b in zip(offline['reviewed_diagnostics'],realtime['reviewed_diagnostics']):
        assert a['id']==b['id']
        md.append(f"| {a['folder']} | {a['target']} | {a['verdict']} | {a['recognized']} | {b['recognized']} |")
    md += ['','실시간에서 광고 연도 2026이 미인식인 것은 번호판 필터 차단 효과가 아니다. 원시 OCR 3개가 여전히 적격이고, 이전의 높은 점수 1.0초 관측이 이번 재생에서 폐기됐다. 남은 관측의 점수·반복 조건으로 보류된 결과다.', '', f"다른 목표 검사: offline {offline['alternative_cases']:,}조합 중 {offline['alternative_passed']}조합 통과, realtime {realtime['alternative_cases']:,}조합 중 {realtime['alternative_passed']}조합 통과. 두 모드의 새 통과 비교 기준은 이전 offline OCR 재판정의 전체 통과 224조합(그중 단일 추가 통과 167조합)으로 고정했다. 시험 목표 목록은 이전·현재 OCR 문자열의 합집합을 사용해 필터로 사라진 숫자도 평가에 남긴다. 실제 다른 버스와 별칭 중복이 포함되므로 이 숫자를 오인식 수로 세지 않는다.",
           f"기존 대비 새로 통과한 다른 목표: offline {len(offline['new_alternative_flags'])}조합, realtime {len(realtime['new_alternative_flags'])}조합. 새 통과는 아래 crop별로 실제 다른 버스/오독 여부를 검토했다.",'',
           '- 전체 출력의 정답 정확도·탑승 안전도 또는 일반화 성능을 의미하지 않는다. 같은 장면을 포함한 개발용 143개에 대한 목표 번호 인식률이다.',
           '- 위치·색상 및 전화번호 형식의 공통 규칙으로 확인된 오류를 보완한 것이다. 백색 번호판, 색상이 다른 차체 차량번호, 연도 광고, 일반 차량 귀속 오류, 손상 LED/부분 번호 오독을 포괄하는 전용 검출기는 아니다.',
           '- 도착 API, 선명도 기반 프레임 선택, 선행 0 별칭, CRAFT 문자 합치기/부분 번호 복원은 이번 변경에 포함하지 않았다.', '',
           '## 재현과 산출물','',
           '```bash', '.venv/bin/python -m unittest discover -s scripts/tests -v',
           '.venv/bin/python scripts/infer_target_video.py --videos-csv output/plate_filter_20260929_run01/inputs.csv --target-from-manifest --single-score 0.98 --sample-fps 5 --output output/새_폴더/offline --device cuda',
           '.venv/bin/python scripts/infer_target_video.py --videos-csv output/plate_filter_20260929_run01/inputs.csv --target-from-manifest --single-score 0.98 --sample-fps 5 --output output/새_폴더/realtime --device cuda --realtime', '```','',
           '검사 35개 통과. 실제 crop 회귀 fixture에는 두 등록번호판과 보호 대상 측면 근거를 포함한다. 원본/정답 CSV/기존 실행은 수정하지 않았다. 상세 CSV, 설정, 환경, 문맥 OCR, 프레임별 지연, 판정 로그는 output/plate_filter_20260929_run01 아래에 보존한다. 사용자 진입점은 workbench/index.html이다.','']
    additions=['## 새 통과의 실제 crop 판독','',
               '| 모드 | 영상 | 목표 | 시각 | 육안 판독 |','|---|---|---|---:|---|']
    for r in evidence:
        if r['kind'] in {'new_alternative','gain','loss'} and r['id']>=30:
            additions.append(f"| {r['mode']} | {r['folder']} | {r['target']} | {r['first_s']} | {r['review_note']} |")
    additions += ['', '10/마포10은 같은 물리적 한글 오독을 별칭으로 중복 시험한 것이다. 이 새 오류와 기존 선택 오류의 잔존을 인식률 증가와 분리해서 해석해야 한다. 색상 기반 제외로 후보 수 초과 또는 경쟁 번호 보류가 해제될 때 올바른 번호와 오독이 모두 새로 통과할 수 있다.', '']
    md += additions
    report='\n'.join(md)
    (WB/'REPORT.md').write_text(report)
    def table(rows):
        return '<table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows[1:])+'</tbody></table>'
    header=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>번호판 제외 검증</title><style>body{{font:16px system-ui;line-height:1.65;margin:24px auto;max-width:1120px;padding:0 16px;background:#f4f6f8;color:#17222d}}section,article{{background:white;padding:18px;margin:18px 0;border-radius:10px;border:1px solid #dce2e8}}table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{border:1px solid #ccc;padding:8px;text-align:left;overflow-wrap:anywhere}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}img.context{{width:100%;max-height:540px;object-fit:contain}}img.crop{{max-width:100%;min-height:45px;max-height:130px;image-rendering:auto}}summary{{cursor:pointer;font-weight:bold}}a{{color:#125bb5}}.muted{{color:#586574;font-size:14px}}</style><a href="../index.html">← 고정 시작 페이지</a><h1>번호판·광고 전화번호 제외 검증</h1><section><p>전체 재추론 <b>{offline['new_recognized']}/143</b>, 원래 속도 재생 <b>{realtime['new_recognized']}/143</b>. 비교 기준은 기존 단일 관측 포함 105/143입니다.</p>{table(metrics)}<p>실시간 처리 p95 <b>{rt['pipeline_ms']['p95']:.2f}ms</b> · 프레임 나이 p95 <b>{rt['frame_age_ms']['p95']:.2f}ms</b>, 최대 <b>{rt['frame_age_ms']['max']:.2f}ms</b> · 폐기 <b>{realtime['dropped_samples']}개</b>.</p><p><a href="REPORT.md">상세 보고서·남은 오류·느린 장면</a> · <a href="../../output/plate_filter_20260929_run01/offline_comparison.csv">offline 전체 CSV</a> · <a href="../../output/plate_filter_20260929_run01/realtime_comparison.csv">realtime 전체 CSV</a></p><p>7284·1577·9350은 두 모드 모두 노선 근거에서 제외됐고 정상 측면 근거는 유지됐습니다. 다만 새로 통과한 10/마포10·20의 3조합(물리적 2사례)은 한글 오독입니다. 목표 번호가 인식된 영상 비율이며 모든 출력의 정답 정확도는 아닙니다. 아래에 기존 오류·보호 대상·새 통과의 실제 crop을 공개합니다.</p></section><section><h2>원본 대조로 바로잡은 출처</h2><p>1577은 뒤쪽 7611 버스 광고 전화번호의 일부, 9350은 그림자 속 후면 등록번호판입니다. 주황색은 버스 검출 박스, 빨간색은 실제 OCR 위치입니다.</p></section>'''
    cards=[]
    for r in evidence:
        prefix='../../output/plate_filter_20260929_run01/evidence/'
        statuses=[]
        for mode in ['offline','realtime']:
            if mode+'_anchor' in r:
                status=', '.join(reason_name.get(p['rejection_reason'],p['rejection_reason']) for p in r[mode+'_anchor']) or '해당 시각 원시 일치 없음'
                statuses.append(mode+': '+status)
        cards.append(f'''<article><h3>{r['id']} · {escape(r['folder'])} · {escape(r['target'])} · {r['first_s']}초</h3><p>{escape(r['review_note'])}</p><p class="muted">{escape(' / '.join(statuses))}</p><img class="context" loading="lazy" src="{prefix+r.get('attribution',r['context'])}"><p><img class="crop" loading="lazy" src="{prefix+r['crop']}"></p><p>원시 OCR: {escape(r['ocr_text'])} · {escape(r['verdict'])}</p><a href="../../{escape(r['video_path'])}#t={r['first_s']}">원본 영상 해당 시각</a></article>''')
    sections=[]
    for title,subset in [('확인된 번호판·전화번호',[i for i,r in enumerate(evidence) if r['id'] in [19,24,25]]),
                          ('보호 대상 정상 노선 근거',[i for i,r in enumerate(evidence) if r['kind']=='gain' and r['id']<26]),
                          ('추가한 측면 회귀 보호 검사',[i for i,r in enumerate(evidence) if r['kind']=='regression_guard']),
                          ('새로 통과하거나 달라진 결과',[i for i,r in enumerate(evidence) if r['id']>=30]),
                          ('다른 실제 버스와 남은 오류',[i for i,r in enumerate(evidence) if r['kind']=='alternative' and r['id'] not in [19,24,25]])]:
        sections.append('<section><h2>'+title+'</h2><div class="grid">'+''.join(cards[i] for i in subset)+'</div></section>')
    rows=[['영상','목표','이전 offline / realtime','이번 offline / realtime','원본 / 결과']]
    offrows=read(OUT/'offline_comparison.csv');rtrows=read(OUT/'realtime_comparison.csv')
    videotable='<table><tr><th>영상</th><th>목표</th><th>offline 이전→현재</th><th>realtime 이전→현재</th><th>영상</th></tr>'
    for a,b in zip(offrows,rtrows):
        videotable+=f'''<tr><td>{escape(a['folder'])}<br>{escape(Path(a['video_path']).name)}</td><td>{escape(a['target'])}</td><td>{a['old_recognized']} → {a['new_recognized']}</td><td>{b['old_recognized']} → {b['new_recognized']}</td><td><a href="../../{escape(a['video_path'])}">원본</a> · <a href="../../output/plate_filter_20260929_run01/realtime/{a['folder']}/annotated.mp4">처리 결과</a></td></tr>'''
    html=header+''.join(sections)+'<details><summary>143개 전체와 반례 1개</summary>'+videotable+'</table></details></html>'
    (WB/'review.html').write_text(html)
    index=ROOT/'workbench/index.html';text=index.read_text()
    new=f'''<section id="current-results"><h2>번호판·광고 전화번호 제외 · 2026-09-29</h2><p>전체 143개: 원본 재추론 <b>{offline['new_recognized']}/143 ({offline['new_recognized']/143:.1%})</b>, 실제 실시간 재생 <b>{realtime['new_recognized']}/143 ({realtime['new_recognized']/143:.1%})</b>. 기존은 각각 105/143.</p><p>7284·1577·9350의 원시 OCR과 노선 근거 허용을 분리 검사했습니다. 정상 측면 PRINT와 남은 광고·부분 번호 오류를 함께 공개합니다.</p><p>프레임 나이 p95 {rt['frame_age_ms']['p95']:.1f}ms / 최대 {rt['frame_age_ms']['max']:.1f}ms · 샘플 폐기 {realtime['dropped_samples']}개.</p><p><a href="plate_filter_20260929_run01/review.html"><b>이번 검증 결과와 실제 crop 보기</b></a></p></section>'''
    text,n=re.subn(r'<section id="current-results">.*?</section>',new,text,count=1,flags=re.S);assert n==1
    if 'single_20260929_run01/review.html' not in text:
        text=text.replace('<summary>보존한 분석·이전 기록</summary><ul>','<summary>보존한 분석·이전 기록</summary><ul><li><a href="single_20260929_run01/review.html">이전 단일 관측 추가 실험</a></li>')
    index.write_text(text)
    files=['core/bus_route/route_evidence.py','scripts/infer_target_video.py',
           'scripts/tests/test_route_evidence.py','scripts/tests/test_target_video.py',
           'output/plate_filter_20260929_run01/inputs.csv']
    provenance={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Published',WB)


if __name__=='__main__':main()
