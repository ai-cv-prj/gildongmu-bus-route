"""Read-only aggregation of latest runs; excludes inactive videos and negative controls."""
import csv,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out=ROOT/'output/combined_20260929_run01';out.mkdir(exist_ok=True)
rows=[]
for dataset,run in [('dataset1','dataset1_target_20260929_run01'),('dataset2','target_20260929_run02')]:
 active={r['video_path']:r for r in read(ROOT/f'data/{dataset}_active.csv')};seen=set()
 for i,s in enumerate(json.loads((ROOT/'output'/run/'summary.json').read_text()),1):
  if s['video'] not in active or s['target_route']!=active[s['video']]['target_route']:continue
  assert s['video'] not in seen;seen.add(s['video'])
  folder=f'video_{i:02d}';ds=read(ROOT/'output'/run/folder/'decisions.csv')
  matched=s['first_matched_candidate_s'] is not None;raw=s['first_exact_s'] is not None
  assert not matched or raw
  conflict=[d for d in ds if d['state']=='conflict']
  category='matched' if matched else 'raw_but_held' if raw else 'no_exact'
  rows.append(dict(dataset=dataset,video_path=s['video'],target=s['target_route'],category=category,matched=matched,raw_exact=raw,conflict_samples=len(conflict),conflicting_numbers=';'.join(sorted({v for d in conflict for v in json.loads(d['conflicts'])})),states=';'.join(sorted({d['state'] for d in ds})),run=run,folder=folder))
 assert seen==set(active),(dataset,set(active)-seen)
write(out/'videos.csv',rows)
routes=[]
for route in sorted({r['target'] for r in rows}):
 group=[r for r in rows if r['target']==route];c=collections.Counter(r['category'] for r in group)
 routes.append(dict(route=route,total=len(group),matched=c['matched'],raw_but_held=c['raw_but_held'],no_exact=c['no_exact'],not_matched=len(group)-c['matched'],not_matched_pct=round(100*(len(group)-c['matched'])/len(group),1)))
routes.sort(key=lambda r:(-r['not_matched_pct'],-r['total'],r['route']));write(out/'routes.csv',routes)
summary={}
for dataset in ['dataset1','dataset2','combined']:
 group=[r for r in rows if dataset=='combined' or r['dataset']==dataset];c=collections.Counter(r['category'] for r in group)
 summary[dataset]=dict(total=len(group),**c,raw_held_with_conflict=sum(r['category']=='raw_but_held' and r['conflict_samples']>0 for r in group),raw_held_without_conflict=sum(r['category']=='raw_but_held' and r['conflict_samples']==0 for r in group),unique_routes=len({r['target'] for r in group}))
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2));print('routes:',', '.join(sorted(r['route'] for r in routes)));print('ranking n>=2:');print(json.dumps([r for r in routes if r['total']>=2],ensure_ascii=False,indent=2))
import html
work=ROOT/'workbench/combined_20260929_run01';work.mkdir(exist_ok=True)
md='''# dataset1 + dataset2 통합 집계 — 2026-09-29

현재 활성 목록 기준 143개 영상(dataset1 79개 + dataset2 64개), 서로 다른 노선 71개입니다. dataset1 공항버스 6002와 dataset2 미정차 제외 6개, 반례 시험은 제외합니다. 기존 평가 로그를 재집계했으며 이번 집계에서 모델을 재실행하거나 정책을 바꾸지 않았습니다. 영상 1개를 1건으로 세며 같은 차량·장면의 중복을 제거한 수는 아닙니다.

| 분류 | dataset1 | dataset2 | 합계 | 비율 |
|---|---:|---:|---:|---:|
| 반복·점수·충돌 조건 통과 | 48 | 45 | 93 | 65.0% |
| 전체 번호 관측, 최종 보류 | 22 | 9 | 31 | 21.7% |
| 전체 목표 문자열 미관측 | 9 | 10 | 19 | 13.3% |
| 전체 | 79 | 64 | 143 | 100% |

31개 보류 중 충돌 기록이 함께 있는 영상은 7개(전체 4.9%), 충돌 기록 없이 보류된 영상은 24개(전체 16.8%)입니다. 충돌 기록이 있다는 것만으로 충돌이 유일한 실패 원인임을 뜻하지 않습니다. 124개(86.7%)는 전체 목표 문자열이 한 번 이상 나왔습니다.

통과는 현재 코드의 후보 판정 통과이며, 사람이 모든 위치를 검수해 확정한 정확도나 오탑승 안전성을 뜻하지 않습니다. 원문 관측은 필터로 제외된 OCR도 포함합니다. 미관측 19개에는 030/30, 072/72, 085/85 같은 표기 차이와 M 분리도 포함되므로 모두 OCR 모델이 숫자를 못 읽었다는 뜻은 아닙니다. 한글 접두어만 기존 규칙으로 허용하며 영어 M/G나 선행 0을 이번에 임의로 없애지 않았습니다. 정차 버스 전체 성공률은 정차 여부가 모두 분류되지 않아 산출하지 않습니다.

## 노선별 현황

버스 종류는 현재 라벨의 노선번호 기준입니다. 차량 제조사·차종은 라벨에 없어 집계하지 않았습니다. 실패율은 (전체−판정 통과)/전체입니다. 표본이 적고 촬영 조건이 달라 실제 운행 환경의 노선별 성능 순위를 뜻하지 않습니다. 특히 M6628 두 영상은 유사 장면입니다. 2개 이상 영상인 노선끼리 우선 비교하고, 1개인 노선은 별도로 봐야 합니다.

| 노선 | 전체 | 통과 | 읽었으나 보류 | 전체 문자열 없음 | 미통과율 |
|---|---:|---:|---:|---:|---:|
'''
for r in routes:md+=f'| {r["route"]} | {r["total"]} | {r["matched"]} | {r["raw_but_held"]} | {r["no_exact"]} | {r["not_matched_pct"]}% |\n'
md+='\n## 번호 관측 후 보류되었고 충돌 기록도 있는 7개\n\n충돌 번호는 당시 OCR 출력이며 실제 버스 번호라고 확정한 것이 아닙니다.\n\n| 영상 | 목표 | 충돌 OCR |\n|---|---|---|\n'
for r in rows:
 if r['category']=='raw_but_held' and r['conflict_samples']:md+=f'| {r["video_path"].removeprefix("data/")} | {r["target"]} | {r["conflicting_numbers"]} |\n'
(work/'REPORT.md').write_text(md,encoding='utf-8')
body='<p>활성 영상 <b>143개</b> · 노선 <b>71개</b> · 판정 통과 <b>93개(65.0%)</b> · 번호 관측 후 보류 <b>31개(21.7%)</b> · 전체 목표 문자열 미관측 <b>19개(13.3%)</b></p><p>번호 관측 후 보류 31개 중 충돌 기록 동반 7개, 충돌 기록 없음 24개. 통과는 코드의 후보 판정이며 검수된 정답 정확도와 다릅니다.</p><p><a href="REPORT.md">정의·제외 기준·충돌 상세</a> · <a href="../../output/combined_20260929_run01/routes.csv">노선별 CSV</a> · <a href="../../output/combined_20260929_run01/videos.csv">영상별 CSV</a></p><h2>노선별 결과</h2><p>미통과율 내림차순, 동률은 표본 수 순. 1개 영상의 실패를 일반적인 성능으로 해석하지 마세요.</p><table><tr><th>노선</th><th>영상 수</th><th>통과</th><th>읽었으나 보류</th><th>전체 문자열 없음</th><th>미통과율</th></tr>'
for r in routes:body+='<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ['route','total','matched','raw_but_held','no_exact','not_matched_pct'])+'</tr>'
body+='</table><h2>영상별 결과</h2>'
labels={'matched':'판정 통과','raw_but_held':'전체 번호 관측 후 보류','no_exact':'전체 문자열 미관측'}
for r in rows:body+=f'<p><a href="../../{html.escape(r["video_path"])}">{html.escape(r["video_path"].removeprefix("data/"))}</a> — {labels[r["category"]]}'+(f' · 충돌 OCR: {r["conflicting_numbers"]}' if r['conflict_samples'] else '')+'</p>'
(work/'review.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>통합 143개 집계</title><style>body{font:16px system-ui;line-height:1.6;max-width:1100px;margin:24px auto;padding:16px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}p{overflow-wrap:anywhere}</style><a href="../index.html">시작 페이지</a><h1>dataset1 + dataset2 통합 결과</h1>'+body+'</html>')
p=ROOT/'workbench/index.html';text=p.read_text();link='<p><a href="combined_20260929_run01/review.html"><strong>dataset1 + dataset2 통합 143개 · 비율과 노선별 순위</strong></a></p>'
if link not in text:text=text.replace('<section><strong>2026-09-29 실행 완료</strong>','<section><strong>2026-09-29 실행 완료</strong>'+link);p.write_text(text)
