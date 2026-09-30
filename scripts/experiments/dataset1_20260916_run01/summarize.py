# 파일명: summarize.py
# 설명: 기존 evaluate.py의 결과를 영상별 CSV와 검토용 HTML/Markdown으로 정리한다.
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/dataset1_20260916_run01'
REPORT_DIR=ROOT/'workbench/dataset1_20260916_run01'
REPORT_DIR.mkdir(parents=True,exist_ok=True)
review=json.loads((OUT/'review.json').read_text())
with (OUT/'predictions.csv').open(encoding='utf-8-sig') as f:
    predictions=list(csv.DictReader(f))
by_id={r['sample_id']:r for r in predictions}
summary=json.loads((OUT/'summary.json').read_text())
readable=[r for r in predictions if r['visibility']=='readable']
comparable=[r for r in predictions if r['filename_label_conflict']=='no']
matches=[r for r in comparable if r['text']==r['target_text_from_filename']]
front=[r for r in readable if r['view']=='front']
side=[r for r in readable if r['view']=='side']
correct=lambda rows:sum(r['exact_match']=='True' for r in rows)
errors=[r for r in readable if r['exact_match']!='True']
allrows=[]
for r in review:
    p=by_id.get(r['sample_id'],{})
    allrows.append(dict(sample_id=r['sample_id'],video_id=r['video_id'],route_id_gt=r['route_id_gt'],
        timestamp_s=r['timestamp_s'],visibility=r.get('visibility','no_roi'),view=r.get('view',''),
        visible_text_gt=p.get('visible_text_gt',''),prediction=p.get('text',''),
        filename_label_conflict=p.get('filename_label_conflict','no'),
        filename_exact_match=(p['text']==p['target_text_from_filename']) if p and p['filename_label_conflict']=='no' else '',
        readable_exact_match=p.get('exact_match',''),bbox_xyxy=json.dumps(r.get('bbox_xyxy')),
        crop_path=r.get('crop_path',''),note=r.get('note','')))
with (OUT/'results_by_video.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)

error_text='\n'.join(f"- `{r['sample_id']}`: `{r['visible_text_gt']}` → `{r['text']}` ({r['video_id']}, {r['timestamp_s']}초)" for r in errors)
report=f'''# dataset1 — 2026-09-16 1차 분석

분석 ID: `dataset1_20260916_run01` · 대상: LED 번호 인식.

## 결과

| 조건 | 일치 / 표본 | 비율 |
|---|---:|---:|
| 사전 육안 검수에서 전체 문자열 판독 가능 | {correct(readable)} / {len(readable)} | {correct(readable)/len(readable):.1%} |
| 위 표본 중 전면 LED | {correct(front)} / {len(front)} | {correct(front)/len(front):.1%} |
| 위 표본 중 측면 LED | {correct(side)} / {len(side)} | {correct(side)/len(side):.1%} |
| 품질 제한 없이 추론한 crop의 파일명 번호 일치(정답 충돌 제외) | {len(matches)} / {len(comparable)} | {len(matches)/len(comparable):.1%} |

LED 포함 파일 63개의 재생 길이 50% 시점에서 각각 한 프레임을 추출했다.
55개 LED 영역에 기존 PARSeq를 실행했다. 8개는 이 시점에 목표 LED 번호 영역을
식별할 수 없어서 crop을 만들지 못했다. 이 8개는 crop 정확도 분모에 들어가지 않는다.
영상 전체에서 판독 가능한 장면이 없다는 뜻은 아니다.

55개 crop의 사전 가시성 분류: readable 30 / partial 19 / no_route 4 / occluded 1 /
정답 충돌로 unverified 1. readable의 정답은 육안 확인한 전체 문자열이다.
나머지의 파일명 일치율은 촬영·표시 상태를 포함한 참고 집계이며, 보이는 문자열의
정답 정확도와 구분한다. 메시지만 표시된 crop에서도 모델은 문자열을 출력했다.

## 실행 조건

- 모델 코드·가중치·전처리·디코딩은 `core/bus_route/recognizer.py` 그대로 사용했다.
- 모델 commit: `{summary['commit']}`.
- GPU: {summary['gpu']}; torch {summary['torch']} / CUDA {summary['cuda_build']}.
- 선택 시각은 예측 전에 고정했다. 좌표와 readable 여부도 모델 실행 전에 지정했다.
- 원본 해상도 crop, 회전 메타데이터 반영, HDR 파일은 기존 코드와 같은 Hable SDR 변환.
- 숫자 필터링·정답 후보 치환·재학습·여러 전처리 중 최선 선택은 하지 않았다.
- bbox는 모델 결과를 본 뒤 수정하지 않았다. 기울기 보정은 적용하지 않았다.
- 원본 영상은 그대로 유지했고 기존 `data/annotations.csv`와 별도 평가 CSV를 사용했다.
- 시점/좌표/가시성 라벨과 crop은 개발 자료이며 독립 블라인드 최종 평가가 아니다.
- 이는 자동 번호 검출, 전체 영상 탐색, 시간 결합, 사건별 사전 안내 성능의 평가가 아니다.
- 63개 파일은 63개 독립 접근 사건으로 검증된 것이 아니다.

## 판독 가능 표본의 오독 3건

{error_text}

## 정답 충돌

`20260916_갤럭시quantum3_272_LED.mp4`의 중간 프레임에는 `271`이 표시되어 있다.
PARSeq도 `271`을 출력했다. 원본 파일명은 수정하지 않았고 정확도 집계에서 제외했다.

## 다시 실행

```bash
.venv/bin/python scripts/evaluate.py \\
  --annotations output/dataset1_20260916_run01/annotations.csv \\
  --output output/dataset1_20260916_run02
```

같은 폴더의 [review.html](review.html): 모든 검토 표본의 이미지·예측 비교 보고서.
`output/dataset1_20260916_run01/results_by_video.csv`: 전체 63개 목록.
`output/dataset1_20260916_run01/predictions.csv`: 원시 추론 결과.
`output/dataset1_20260916_run01/summary.json`: 집계·환경·가중치 해시.
`output/dataset1_20260916_run01/manifest.json`, `review.json`: 시점·좌표·제외 사유.
`scripts/experiments/dataset1_20260916_run01/`: 이번 실험용 좌표·라벨·집계 코드.

평가 CSV SHA-256: `{hashlib.sha256((OUT/'annotations.csv').read_bytes()).hexdigest()}`.

## 결과 정리 내역

분석 보고서를 `workbench/`로, 실험 코드를 `scripts/experiments/`로 이동했다.
`output/`의 `details/parseq_results/` 중첩을 없애고 원시 기록과 이미지를 분석 ID 아래 모았다.
역할이 끝난 초기 2장 CPU/GPU 테스트 기록 `archive/initial_checks/`는 삭제했다.
원본 영상·평가 crop·정답·예측·집계는 보존했으며 모델 재추론은 하지 않았다.
`summary.json`의 `command`는 당시 실행한 명령의 이력으로, 이전 경로가 남아 있을 수 있다.
재실행에는 위의 갱신된 명령을 사용한다.
'''
(REPORT_DIR/'REPORT.md').write_text(report,encoding='utf-8')

def encoded_image(path, width=420, height=200):
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail((width,height));buf=io.BytesIO();im.save(buf,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()

cards=[]
for r in allrows:
    p=by_id.get(r['sample_id'])
    color='good' if r['readable_exact_match']=='True' else 'bad' if r['readable_exact_match']=='False' else 'unknown'
    source=ROOT/r['crop_path'] if r['crop_path'] else Path(next(x['frame_path'] for x in review if x['sample_id']==r['sample_id']))
    tag='crop' if r['crop_path'] else '전체 프레임: 번호 ROI 없음'
    cards.append(f'''<article class="{color}"><h3>{html.escape(r['sample_id'])} · {html.escape(r['visibility'])}</h3>
<p class="filename">{html.escape(r['video_id'])}</p><p>{r['timestamp_s']}초 · {html.escape(r['view'])} · {tag}</p>
<img src="{encoded_image(source)}" alt="{tag}"><p>파일명 노선: <b>{html.escape(r['route_id_gt'])}</b></p>
<p>보이는 문자열 정답: <b>{html.escape(r['visible_text_gt']) or '미확정'}</b><br>PARSeq: <b>{html.escape(r['prediction']) or ('빈 문자열' if p else '실행 안 함')}</b></p>
<p>{html.escape(r['note'])}</p></article>''')
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>dataset1 · 20260916 · 1차 분석</title>
<style>body{{font-family:system-ui,sans-serif;margin:28px;background:#f5f6f8;color:#17202a}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:16px}}article{{background:white;padding:18px;border-radius:12px;border-top:5px solid #aaa}}.good{{border-color:#21845b}}.bad{{border-color:#b83939}}.filename{{overflow-wrap:anywhere;font-size:13px}}img{{max-width:100%;min-height:35px;object-fit:contain;image-rendering:auto}}p{{line-height:1.5}}</style>
<h1>dataset1 · 2026-09-16 · 1차 분석</h1><p>분석 ID: dataset1_20260916_run01 · LED 번호 crop의 PARSeq 추론 결과</p><p>사전 육안 판독 가능 <b>{correct(readable)}/{len(readable)} ({correct(readable)/len(readable):.1%})</b> · 품질 제한 없는 crop의 파일명 일치 <b>{len(matches)}/{len(comparable)} ({len(matches)/len(comparable):.1%})</b>.</p>
<p>63개 파일의 중간 시점 한 프레임씩 검토 → 55개 영역 추론 / 8개 ROI 없음. 정답 충돌 1개 제외.
전체 영상 성능이나 버스 접근 사건 성공률이 아닙니다. 원본 해상도로 추론했으며 아래 이미지는 보기용 크기입니다.</p>
<p>초록: readable 정답 · 빨강: readable 오독 · 회색: 판독 불완전/번호 미표시/정답 충돌/ROI 없음.</p>
<div class="grid">{''.join(cards)}</div></html>'''
(REPORT_DIR/'review.html').write_text(page,encoding='utf-8')
print(report)
