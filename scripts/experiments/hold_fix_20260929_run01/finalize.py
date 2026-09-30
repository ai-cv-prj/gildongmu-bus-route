import csv,json,html,hashlib,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3];O=R/'output/hold_fix_20260929_run01';W=R/'workbench/hold_fix_20260929_run01';W.mkdir(exist_ok=True)
s=json.loads((O/'comparison_summary.json').read_text());rt=json.loads((O/'realtime/summary.json').read_text())
ages=[]
for f in (O/'realtime').glob('video_*/frames.csv'):
 with f.open(encoding='utf-8-sig') as stream:ages.extend(float(x['frame_age_ms']) for x in csv.DictReader(stream))
s['realtime']={'videos':len(rt),'processed_frames':sum(x['sample_count'] for x in rt),'dropped_frames':sum(x['dropped_samples'] for x in rt),'frame_age_p95_ms':float(np.percentile(ages,95)),'frame_age_max_ms':max(ages),'matched_videos':sum(x['first_matched_candidate_s'] is not None for x in rt)}
(O/'comparison_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
with (O/'comparison.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
evidence=json.loads((O/'evidence/records.json').read_text());verdicts=json.loads((O/'evidence/visual_review.json').read_text())
matched=s['new_match'];raw=s['new_raw'];held=raw-matched;missing=143-raw
md=f'''# 보류 처리 수정 결과 — 2026-09-29

같은 활성 영상 143개를 재추론했습니다. 후보 판정 통과는 **{s['old_match']}/143 → {matched}/143 ({matched/143:.1%})**입니다. 신규 통과 {len(s['gains'])}개, 기존 통과에서 보류로 바뀐 영상 {len(s['losses'])}개입니다. 종전 ‘읽었으나 보류’ 31개 중 {s['prior31_recovered']}개가 통과했습니다.

| 현재 결과 | 영상 수 | 비율 |
|---|---:|---:|
| 판정 통과 | {matched} | {matched/143:.1%} |
| 전체 목표 문자열 관측 후 보류 | {held} | {held/143:.1%} |
| 전체 목표 문자열 미관측 | {missing} | {missing/143:.1%} |

이 수치는 현재 코드 조건을 통과한 영상 비율이며 독립 검증 정확도나 오탑승 확률이 아닙니다. 제외된 영상과 반례 영상은 분모 143에 넣지 않았습니다. 모델이나 OCR 점수·반복 횟수·경쟁 번호 기준을 바꾸지 않았습니다.

## 수정 내용

- 상단/차체 탐색이 겹치면 기존 상단 후보가 잘렸어도 차체 후보가 무조건 버려지던 처리를 수정했습니다. 기존 후보는 부적격이고 새 후보는 적격인 경우에만 교체합니다. 목표 번호는 후보 선택에 사용하지 않습니다.
- crop 검사에서 이미 부적격인 후보보다 적격 후보를 OCR 예산에 우선 배정합니다. 미처리 적격 후보가 남아 있을 때만 후보 제한으로 판정을 보류합니다. 미처리 적격 경쟁 번호가 있을 가능성은 계속 보수적으로 처리합니다.
- 모델, 3시점 반복, 1.2초 창, 최소 점수 0.9, 강한 경쟁 번호 거부, 부분 숫자 추정 금지, 번호판·폭 필터는 유지했습니다. 미정차 영상은 다시 포함하지 않았습니다.

## 검증과 한계

- 단위·통합 테스트 28개 통과. 잘린 상단/온전한 차체 중복, 부적격 후보가 예산을 차지하는 경우, 미처리 적격 후보가 있으면 보류하는 경우를 검증했습니다.
- 전체 143개 및 실제 7017 영상에 목표 7016을 준 반례 1개 재추론. 반례 통과 여부: {s['negative_7017_as_7016_passed']}.
- 다른 목표 {s['alternative_cases']}조합을 기존/수정 OCR 로그에 같은 조건으로 재생했습니다. 수정 후 통과한 다른 목표 조합 {s['alternative_flags']}개, 기존에는 통과하지 않았던 조합 {s['new_alternative_flags']}개입니다. 화면에 다른 버스가 함께 나올 수 있으므로 이 수를 곧바로 오확정 수로 해석하지 않습니다. 아래 육안 검토를 참조하세요.
- offline 처리 시간 p50 {s['pipeline_p50_ms']:.1f}ms, p95 {s['pipeline_p95_ms']:.1f}ms. 원래 속도 재생 {s['realtime']['videos']}개에서 처리 {s['realtime']['processed_frames']}프레임, 폐기 {s['realtime']['dropped_frames']}프레임, 판단 시점 프레임 나이 p95 {s['realtime']['frame_age_p95_ms']:.1f}ms. 휴대폰 촬영·전송·음성 안내 전체 지연을 포함하지 않습니다.
- 기존 광고 한글을 숫자로 오독하는 한계는 이번 수정 대상이 아닙니다. 탑승 확정에 바로 사용할 검증은 아닙니다.
- 31개 보류의 로그 근거는 prior_31_hold_reasons.json입니다. LED 손상·반사·검출·필터는 함께 작용할 수 있어 자동 판정 상태만으로 광학 원인별 비중을 확정하지 않았습니다.
- 실제 추적 ID를 요구하면 31개 중 28개는 같은 버스에서 높은 점수의 유효한 전체 번호가 1.2초 안에 3회 모이지 않았습니다. 나머지 3개도 충돌 또는 후보 제한으로 보류됐습니다. 앞선 중간 집계 27개는 추적 ID가 없는 7713 영상의 관측을 하나의 트랙으로 묶은 값이어서 최종 집계에서 바로잡았습니다.
- 후보 제한으로 인한 차단은 10개 버스·시점에서 해소됐고, 4개 영상에서 높은 점수의 유효 목표 관측이 총 6시점 늘었습니다. 기존 유효 목표 관측이 사라진 시점은 없습니다. 보류 1200 영상은 유효 근거가 0→1시점으로 늘었지만 3시점을 채우지 못해 계속 보류됐습니다. candidate_effects.json 참조.
- 다른 목표에서 통과한 기존 57개 조합 전체를 이번에 새로 육안 검수하지는 않았습니다. 신규 통과 조합이 0이라는 검사는 기존 오류가 해결됐다는 뜻이 아닙니다.

## 신규 판정의 영상 근거 검토

'''
for v in verdicts:md+=f'- {v["video_path"].removeprefix("data/")} / 목표 {v["target"]}: {v["note"]}\n'
if not verdicts:md+='새로 통과한 영상 또는 다른 목표 조합이 없어 신규 판정 crop 검토 대상은 없습니다.\n'
md+='\n## 다음 우선 작업\n\n이번은 후보 선택에서 정상 근거가 사라지는 두 경로만 수정한 첫 단계입니다. 번호 영역 전체 검출, 비스듬한 측면 번호의 폭 필터, 노선별 표기 별칭은 별도 검증이 필요합니다. 점수/반복 기준을 일괄 완화하거나 깨진 숫자를 목표에 맞춰 채우지 않습니다. 추가 촬영보다 기존 보류 영상의 번호 영역과 시간별 근거를 먼저 보완합니다.\n'
(W/'REPORT.md').write_text(md)
body=f'<p>판정 통과 <b>93/143 → {matched}/143 ({matched/143:.1%})</b>. 신규 통과 {len(s["gains"])}개, 통과→보류 {len(s["losses"])}개. 코드의 후보 판정 비율이며 정답 검수 정확도와 다릅니다.</p><p><a href="REPORT.md">상세 보고서</a> · <a href="../../output/hold_fix_20260929_run01/comparison.csv">전체 비교 CSV</a></p>'
for r,v in zip(evidence,verdicts):
 assert r['video_path']==v['video_path'] and r['target']==v['target']
 body+=f'<article><h2>{html.escape(Path(r["video_path"]).name)} / 목표 {r["target"]}</h2><p>{html.escape(v["note"])}</p><a href="../../{html.escape(r["video_path"])}">원본</a> · <a href="../../output/hold_fix_20260929_run01/{r["folder"]}/annotated.mp4">처리 영상</a><p><img src="../../output/hold_fix_20260929_run01/evidence/{r["context"]}"><img src="../../output/hold_fix_20260929_run01/evidence/{r["crop"]}"></p></article>'
body+='<h2>전체 143개 결과</h2><table><tr><th>영상</th><th>수정 전</th><th>수정 후</th></tr>'
for r in rows:body+=f'<tr><td><a href="../../{html.escape(r["video_path"])}">{html.escape(Path(r["video_path"]).name)}</a></td><td>{"통과" if r["old_match"]=="True" else "보류"}</td><td>{"통과" if r["new_match"]=="True" else "보류"}</td></tr>'
(W/'review.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보류 처리 수정 결과</title><style>body{font:16px system-ui;line-height:1.6;max-width:1100px;margin:24px auto;padding:16px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}img{max-width:48%;max-height:600px}article{border-top:1px solid #ccc;padding:16px 0}</style><a href="../index.html">시작 페이지</a><h1>보류 처리 수정 결과</h1>'+body+'</table></html>')
p=R/'workbench/index.html';text=p.read_text();anchor='<h1>지금은 이 페이지 하나만 보세요</h1>';block=f'<section><strong>최신: 보류 처리 수정 및 전체 재평가 완료</strong><p>판정 통과 93/143 → {matched}/143 ({matched/143:.1%}). <a href="hold_fix_20260929_run01/review.html">수정 결과와 영상 근거</a></p><p>아래 통합 143개 집계와 기존 확인 도구는 수정 전 기록입니다.</p></section>'
if block not in text:text=text.replace(anchor,anchor+block);p.write_text(text)
for source,name in [('/tmp/bus_hold_fix_realtime.log','realtime.log')]:shutil.copyfile(source,O/name)
(O/'source_sha256.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'scripts/infer_target_video.py',*sorted((R/'core/bus_route').glob('*.py')),*sorted(Path(__file__).parent.glob('*.py'))]},indent=2))
print('Final report complete',matched,'/143')
