"""Produce one final report for offline policy replay + all-video realtime inference."""
import csv,json,html,hashlib,shutil,re,collections
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3];O=R/'output/single_20260929_run01';W=R/'workbench/single_20260929_run01';W.mkdir(exist_ok=True)
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
s=json.loads((O/'offline/summary.json').read_text());rt=json.loads((O/'realtime/summary.json').read_text());assert len(rt)==144
off=read(O/'offline/comparison.csv');rows=[];ages=[];pipeline=[];stale=0
for i,x in enumerate(rt,1):
 f=O/'realtime'/f'video_{i:02d}';frames=read(f/'frames.csv');decisions=read(f/'decisions.csv')
 assert len(frames)==x['sample_count']
 recognized=[d for d in decisions if d['state'] in {'recognized_single','matched_candidate'}]
 repeat=[d for d in decisions if d['state']=='matched_candidate']
 assert bool(recognized)==(x['first_recognized_s'] is not None)
 assert bool(repeat)==(x['first_matched_candidate_s'] is not None)
 if i<=143:
  ages.extend(float(a['frame_age_ms']) for a in frames);pipeline.extend(float(a['pipeline_ms']) for a in frames)
  stale+=sum(float(a['frame_age_ms'])>300 for a in frames)
 rows.append(dict(video_path=x['video'],target=x['target_route'],folder=f'video_{i:02d}',offline_old=off[i-1]['old_recognized']=='True',offline_new=off[i-1]['new_recognized']=='True',realtime_old=bool(repeat),realtime_new=bool(recognized),first_recognized_s=x['first_recognized_s'],first_repeated_s=x['first_matched_candidate_s'],sample_count=x['sample_count'],dropped_samples=x['dropped_samples'],test_kind='positive_target' if i<=143 else '7017_as_7016_negative'))
positive=rows[:143];rn=sum(r['realtime_new'] for r in positive);ro=sum(r['realtime_old'] for r in positive);pairs=[r['first_repeated_s']-r['first_recognized_s'] for r in positive if r['realtime_old']]
verdicts=json.loads((O/'evidence/visual_review.json').read_text());negative_counts=collections.Counter(r['verdict'] for r in verdicts if r['kind']=='alternative')
metrics=dict(videos=143,offline_old=93,offline_new=s['new_recognized'],realtime_old=ro,realtime_new=rn,realtime_gains=sum(r['realtime_new'] and not r['realtime_old'] for r in positive),realtime_losses=sum(r['realtime_old'] and not r['realtime_new'] for r in positive),realtime_vs_offline_losses=[r for r in positive if r['offline_new'] and not r['realtime_new']],realtime_vs_offline_gains=[r for r in positive if not r['offline_new'] and r['realtime_new']],processed_frames=sum(r['sample_count'] for r in positive),dropped_samples=sum(r['dropped_samples'] for r in positive),frame_age_p50_ms=float(np.percentile(ages,50)),frame_age_p95_ms=float(np.percentile(ages,95)),frame_age_max_ms=max(ages),frame_age_over_300ms_samples=stale,pipeline_p50_ms=float(np.percentile(pipeline,50)),pipeline_p95_ms=float(np.percentile(pipeline,95)),earlier_than_repetition_median_s=float(np.median(pairs)) if pairs else None,negative_control_offline=s['negative_control']['new_recognized'],negative_control_realtime=rows[-1]['realtime_new'],alternative_cases=s['alternative_cases'],new_alternative_flags=s['new_alternative_flags'],reviewed_alternative_cases=dict(negative_counts),visual_review_scope='all 12 offline gains; 14 selected alternative combinations, including one alias duplicate; not random or exhaustive')
assert not metrics['realtime_losses']
write(O/'comparison.csv',rows);(O/'summary.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
md=f'''# 단일 관측 인식 시험 — 2026-09-29

**전체 143개에서 저장된 OCR 기준 93개(65.0%) → {s['new_recognized']}개({s['new_recognized']/143:.1%})로 12개 증가했습니다. 실제 원래 속도 재생에서는 기존 반복 기준 {ro}개({ro/143:.1%}) → 단일 관측 포함 {rn}개({rn/143:.1%})입니다.**

## 무엇을 바꿨나

영상 초반·중간·끝에 관계없이 현재 프레임의 전체 목표 번호를 한 번 읽었고 모든 문자와 EOS 점수가 0.98 이상이면 recognized_single로 기록합니다. 기존 3회 반복 기준은 matched_candidate로 따로 유지합니다. 두 상태 모두 ‘번호 인식’ 집계에 포함하되 단일 관측이 반복 확인과 같은 수준의 검증이라는 뜻은 아닙니다. 완전한 문자열 비교, 한글 지역 접두어 생략, 추적 ID 필요, crop/번호판 필터, 1.2초 이내 경쟁 번호, 후보 처리 완전성, 실시간 300ms 지연 제한을 유지합니다. 과거에 본 번호를 현재 프레임의 관측처럼 유지하지 않습니다.

기본 실행은 단일 점수 0.98을 사용합니다. --repeat-only로 기존 판정으로 되돌릴 수 있습니다. 모델·문자 보정·탐색 영역·샘플링 5fps는 변경하지 않았습니다. ‘선명한 프레임 자동 선택’은 아직 구현하지 않았습니다. OCR 점수는 보정된 정답 확률이 아닙니다.

## 결과 해석

| 평가 | 기존 반복만 | 단일 관측 포함 | 증가 |
|---|---:|---:|---:|
| 저장된 전체 OCR를 새 조건으로 재판정 | 93/143 (65.0%) | {s['new_recognized']}/143 ({s['new_recognized']/143:.1%}) | +12개, +8.4%p |
| 전체 영상을 원래 속도로 새로 추론 | {ro}/143 ({ro/143:.1%}) | {rn}/143 ({rn/143:.1%}) | +{rn-ro}개, +{(rn-ro)/143*100:.1f}%p |

실시간의 ‘기존’은 새 실시간 실행에서 실제 처리된 동일 프레임에 적용된 기존 반복 기준입니다. 서로 다른 프레임 표본을 같은 입력인 것처럼 비교하지 않았습니다. OCR 재판정과 실제 실시간 추론을 구분했습니다. 제외 목록은 유지했고, 7017에 목표 7016을 준 반례 1개는 143개 분모 밖에서 추가 실행했습니다.

추가 인식된 12개 영상의 첫 인식 위치를 모두 확인했습니다. 실제 노선 LED/PRINT의 전체 번호이고 번호판·광고를 읽은 사례는 이 12개에서 발견하지 못했습니다. 원거리나 번짐으로 시각 품질이 낮은 번호도 있어 전체 프레임 정답 전수 검증을 뜻하지는 않습니다. 인식률은 목표가 있는 영상에서 목표 문자열이 인식된 비율이며, 모든 안내의 정확도 또는 오탑승 확률이 아닙니다.

## 단일 관측에서 확인된 문제

같은 OCR 로그에 다른 목표를 넣은 {s['alternative_cases']:,}조합에서 기존에 없던 통과 조합이 {s['new_alternative_flags']}개 생겼습니다. 실제 다른 버스가 함께 나타나거나 같은 숫자 별칭을 중복 시험한 경우도 있어 이 숫자를 오인식 건수로 해석해서는 안 됩니다.

유사 번호 혼동과 번호판·광고 위험을 보기 위해 선택한 14개 조합을 육안 확인했습니다. **12개 조합은 잘못된 노선 인식, 2개는 실제 다른 버스의 노선**이었습니다. 12개 중 1개는 같은 LED를 ‘11’과 ‘마포11’로 시험한 중복으로, 서로 다른 오류 관측은 11개입니다. 선택 검사이므로 이 비율을 서비스 오인식률로 계산하지 않습니다. 나머지 새 통과 조합은 전부 육안 검수하지 않았습니다.

- 마포13의 손상 LED 일부를 11로 읽음(점수 약 0.992).
- 7011 표시에서 70만 검출하여 목표 70으로 인식(약 0.998).
- 등록 번호판 7284·1577, 차량 식별번호 9350을 노선으로 인식.
- 광고 연도 2026과 전면 유리의 한글을 숫자로 인식.
- 배경 버스를 가린 일반 차량의 153 표기를 버스에 귀속.
- 271 영상 초반의 273과 271 옆의 3400은 실제 다른 버스의 표시였음.

**단일 관측으로 목표 발견률은 개선됐지만, 점수 조건만으로 ‘실제 노선 표시 전체를 읽었다’는 것을 보장할 수 없습니다. 따라서 단일 관측을 반복 확인과 구분해야 하며, 이 실험 결과를 곧바로 확정 탑승 안내 성능으로 해석하지 않습니다.**

점수 기준 민감도도 저장된 로그로 비교했습니다. 0.98에서 105/143, 0.99와 0.995에서 각각 104/143이 인식됐습니다. 위에서 확인한 잘못된 12개 조합 중 여전히 통과한 것은 각각 12·11·8개입니다. 선택한 음성 사례의 비교일 뿐 오인식률은 아니며, 단순히 점수만 높여도 출처 오류가 남는다는 근거입니다. 실제 전체 실시간 실행은 사전에 정한 0.98 기준으로 유지했습니다. threshold_sensitivity.json 참조.

## 실시간 처리

RTX 5080 Laptop GPU에서 전체 143개 영상을 5fps로 원래 속도 재생했습니다. 처리 {metrics['processed_frames']:,}프레임, 처리 지연에 따른 샘플 폐기 {metrics['dropped_samples']}개입니다. 판단 프레임 나이 p50 {metrics['frame_age_p50_ms']:.1f}ms, p95 {metrics['frame_age_p95_ms']:.1f}ms, 최대 {metrics['frame_age_max_ms']:.1f}ms입니다. 처리 단계만의 p95는 {metrics['pipeline_p95_ms']:.1f}ms입니다. 판정 후 프레임 나이가 300ms를 넘은 샘플은 {stale}개이며 판정 직전의 최신성 검사는 별도로 적용합니다. 넘었다는 이유만으로 그 샘플에서 인식을 안내했다고 해석하면 안 됩니다.

이는 PC GPU의 파일 재생 시험입니다. 휴대폰 자체 추론, 영상 전송, 실제 카메라 큐와 음성 안내 지연은 포함하지 않았습니다. 짧은 선명 구간이 5fps 표본 사이에 놓이면 여전히 놓칠 수 있습니다. 단일 관측 판정 자체는 추가 모델 추론을 요구하지 않습니다.

평균적인 속도만 보면 놓칠 수 있는 예외가 있습니다. dataset1의 `20260914_갤럭시quantum3_271_LED_빛번짐.mp4`는 여러 버스를 처리하면서 파이프라인 p50 약 211ms로 5fps의 200ms 간격보다 느렸습니다. 프레임 나이가 최대 약 411ms까지 쌓여 300ms 제한으로 반복 근거가 보류됐습니다. 기존 offline에서는 반복 통과했지만 이번 realtime에서는 단일 관측으로만 인식됐습니다. 이 영상은 이전 코드의 offline 처리 p50도 약 212ms였으므로 단일 관측 조건이 만든 모델 부하가 아닙니다. 다중 버스 장면의 검출·OCR 작업량과 처리 지연 상한은 추가 최적화가 필요합니다. 모든 장면에서 5fps 지연 예산을 충족했다고 주장하지 않습니다.

## 검증

- 단위·통합 테스트 31개 통과. 단일 관측의 임의 시각, EOS, 경쟁 번호, 불완전 탐색, 낮은 점수, 현재 프레임 부재, 기존 반복 기준 유지 검사.
- 저장된 144개 로그를 재판정할 때 기존 반복 통과 여부가 모두 동일한지 assertion으로 대조.
- 실제 7017 영상에 목표 7016 지정: offline 인식 {metrics['negative_control_offline']}, realtime 인식 {metrics['negative_control_realtime']}.
- 전체 realtime summary와 판정/프레임 CSV의 행 수·인식 상태 대조.
- 실험에 사용한 영상은 이미 검토한 데이터입니다. 독립 일반화 성능 검증셋은 아닙니다.

## 다음 판단

단일 관측을 활용하는 방향에는 효과가 있습니다. 다음 우선순위는 OCR 점수를 더 낮추는 것이 아니라 노선 표시 영역을 번호판·광고와 구분하고, 일부 숫자만 잡힌 crop을 검증하는 것입니다. 기존 자료에 그런 음성 사례가 있으므로 추가 촬영부터 요구할 필요는 없습니다. 속도는 이 GPU의 5fps 처리에서 실측한 수치로 판단하며, 표시 영역 검증을 추가할 때도 같은 실시간 시험을 반복해야 합니다.

[영상별 비교와 근거 이미지](review.html)
'''
(W/'REPORT.md').write_text(md)
body=f'<section><h2>인식률 변화</h2><p>전체 143개 · 저장 OCR 재판정 <b>65.0% → 73.4%</b> · 실제 실시간 재생 <b>{ro/143:.1%} → {rn/143:.1%}</b></p><p>신규 인식 12개 위치는 실제 노선 표시였습니다. 다만 다른 목표 검사에서 번호판·광고·부분 번호의 잘못된 단일 인식도 확인했습니다. 단일 관측과 반복 확인을 구분합니다.</p><p>프레임 나이 p95 {metrics["frame_age_p95_ms"]:.1f}ms · 샘플 폐기 {metrics["dropped_samples"]}개 · PC GPU, 5fps.</p><a href="REPORT.md">상세 보고서</a> · <a href="../../output/single_20260929_run01/comparison.csv">영상별 CSV</a></section>'
for kind,title in [('gain','새로 인식된 12개 영상'),('alternative','선택한 다른 목표 14개 조합 검토')]:
 body+=f'<h2>{title}</h2>'
 for r in verdicts:
  if r['kind']!=kind:continue
  body+=f'<details><summary>{html.escape(Path(r["video_path"]).name)} · 목표 {html.escape(r["target"])} · {r["first_s"]}초</summary><p>{html.escape(r["review_note"])}</p><p>OCR 점수 {r["score"]:.4f} — 정답 확률이 아님</p><a href="../../{html.escape(r["video_path"])}">원본 영상</a><p><img loading="lazy" src="../../output/single_20260929_run01/evidence/{r["context"]}"><img loading="lazy" src="../../output/single_20260929_run01/evidence/{r["crop"]}"></p></details>'
body+='<h2>전체 영상 비교</h2><p>아래 두 열은 같은 실시간 실행에서 처리한 프레임 기준입니다.</p><table><tr><th>영상</th><th>반복만</th><th>단일 포함</th><th>첫 인식(초)</th></tr>'
for r in positive:body+=f'<tr><td><a href="../../{html.escape(r["video_path"])}">{html.escape(Path(r["video_path"]).name)}</a></td><td>{"인식" if r["realtime_old"] else "보류"}</td><td>{"인식" if r["realtime_new"] else "보류"}</td><td>{r["first_recognized_s"] if r["realtime_new"] else "—"}</td></tr>'
(W/'review.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>단일 관측 시험 — 인식률·오인식·실시간 결과</title><style>body{font:16px system-ui;line-height:1.6;max-width:1100px;margin:24px auto;padding:16px}section,details{border:1px solid #ccc;padding:16px;margin:12px 0}summary{cursor:pointer}img{max-width:48%;max-height:650px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><a href="../index.html">고정 시작 페이지</a><h1>단일 관측 시험 — 2026-09-29</h1>'+body+'</table></html>')
# One stable entry. Move earlier headline results into an explicit archive.
p=R/'workbench/index.html';text=p.read_text();start=text.index('<h1>');end=text.index('<main>',start)
current=f'''<h1>버스 인식 — 현재 시험 결과</h1><section id="current-results"><h2>단일 관측 인식 추가 · 2026-09-29</h2><p>전체 143개: 저장 OCR 기준 <b>65.0% → 73.4%</b>, 실제 실시간 재생 <b>{ro/143:.1%} → {rn/143:.1%}</b>.</p><p>추가 인식 효과는 확인했습니다. 동시에 번호판·광고·부분 번호의 단일 오인식도 확인돼, 단일 관측을 확정 탑승 판단으로 해석해서는 안 됩니다.</p><p>PC GPU에서 5fps · 프레임 나이 p95 {metrics['frame_age_p95_ms']:.1f}ms · 샘플 폐기 {metrics['dropped_samples']}개.</p><p><a href="single_20260929_run01/review.html"><b>이번 시험 결과와 영상 근거 보기</b></a></p></section><details><summary>이전 결과 — 비교용 기록</summary><ul><li><a href="hold_fix_20260929_run01/review.html">후보 선택 수정: 인식률 변화 없음</a></li><li><a href="combined_20260929_run01/review.html">기존 통합 집계: 93/143, 65.0%</a></li><li><a href="diagnosis_20260929_run01/review.html">dataset2 정차 보류 19개 원인</a></li><li><a href="dataset1_target_20260929_run01/review.html">dataset1 기존 전체 평가</a></li></ul></details><details id="review-history"><summary>이전 정차 분류 기록 열기 — 아래 인식 여부는 수정 전 결과</summary>'''
tail=text[end:].replace('</main></details>','</main>',1)
text=text[:start]+current+tail;text=text.replace('</main>','</main></details>',1);text=text.replace('<title>버스 영상 확인 — 여기서 시작</title>','<title>버스 인식 결과 — 고정 시작 페이지</title>');p.write_text(text)
for src,dest in [('/tmp/bus_single_tests.log','tests.log'),('/tmp/bus_single_realtime.log','realtime.log'),('/tmp/bus_single_evaluate.log','offline/evaluate.log')]:shutil.copyfile(src,O/dest)
files=[*sorted((R/'core/bus_route').glob('*.py')),R/'scripts/infer_target_video.py',R/'scripts/tests/test_target.py',R/'scripts/tests/test_target_video.py',R/'README.md',*sorted(Path(__file__).parent.glob('*.py'))]
(O/'source_sha256.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps(metrics,ensure_ascii=False,indent=2))
