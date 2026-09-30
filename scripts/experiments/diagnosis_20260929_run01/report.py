import csv,json,html,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[3]
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def page(title,body):return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>body{font:16px system-ui;line-height:1.6;max-width:1100px;margin:24px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}img{max-width:100%}details{border:1px solid #ddd;padding:12px;margin:12px 0}a{color:#1565c0}</style><a href="../index.html">시작 페이지</a><h1>'+title+'</h1>'+body+'</html>'
out=R/'output/dataset1_target_20260929_run01';work=R/'workbench/dataset1_target_20260929_run01';work.mkdir(exist_ok=True)
s=json.loads((out/'summary.json').read_text());rows=[]
for i,x in enumerate(s,1):
 folder=f'video_{i:02d}';preds=read(out/folder/'predictions.csv');ds=read(out/folder/'decisions.csv')
 assert len(read(out/folder/'frames.csv'))==x['sample_count']
 assert sum(p['state']=='matched_candidate' for p in ds)==x['decision_counts'].get('matched_candidate',0)
 rows.append(dict(folder=folder,video_path=x['video'],target=x['target_route'],raw_exact=x['first_exact_s'] is not None,matched=x['first_matched_candidate_s'] is not None,first_exact_s=x['first_exact_s'],first_matched_s=x['first_matched_candidate_s'],samples=x['sample_count']))
assert len(rows)==79
writecsv(out/'comparison.csv',rows);shutil.copyfile(R/'data/dataset1_active.csv',out/'inputs.csv')
files=[R/'scripts/infer_target_video.py',*sorted((R/'core/bus_route').glob('*.py'))]
(out/'source_sha256.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
shutil.copyfile('/tmp/dataset1_target_20260929.log',out/'execution.log')
raw=sum(x['raw_exact'] for x in rows);matched=sum(x['matched'] for x in rows)
intro=f'''79개 전체를 최신 코드로 GPU 재평가했습니다. **전체 번호가 한 번 이상 OCR에 나온 영상은 {raw}/79 ({raw/79:.1%}), 반복·신뢰도·충돌 조건까지 통과한 영상은 {matched}/79 ({matched/79:.1%})**입니다. 31개는 보류이며, 그중 22개는 번호를 읽은 적이 있지만 최종 조건을 충족하지 못했습니다. 9개는 전체 번호가 나오지 않았습니다.

이는 영상 단위 목표 번호 발견률/판정 통과율입니다. 정답 위치를 프레임별로 검수한 정확도나 잘못된 버스에 대한 오탑승률은 아닙니다. 목표 번호는 manifest에서 제공한 실험이며 실제 도착 API 연결은 없습니다.

- GPU: RTX 5080 Laptop, YOLO11n + CRAFT + PARSeq, 5 fps 샘플링, 총 2,749프레임.
- 최신 dataset2 run02와 같은 설정. 공항버스 6002 제외, 대상 목록 79개 전부 완료.
- 이전 dataset1 원문 OCR 결과를 접두어 규칙으로 재집계한 52/79와 현재 원문 70/79는 비교할 수 있으나, 이전 결과에 현재의 판정 통과율 기준은 없으므로 48/79와 직접 비교하지 않습니다.
- 이번 실행은 offline 전체 처리입니다. 실시간 재생 지연·프레임 누락 성능 검증을 새로 수행한 것은 아닙니다.
- 실패를 줄이기 위해 임계값을 낮추거나 701?를 7016으로 채우지 않았습니다. 모델·판정 코드 변경 없이 최신 구현을 재평가했습니다.
- 79개 프레임 로그 행 수와 summary, 판정 통과 수를 대조했습니다. 파일별 결과는 comparison.csv, 재현 설정·모델 해시는 settings.json, 코드 해시는 source_sha256.json에 있습니다.
'''
(work/'REPORT.md').write_text('# dataset1 전체 재평가 — 2026-09-29\n\n'+intro+'\n[영상별 결과](review.html)\n',encoding='utf-8')
body='<p>'+html.escape(intro.split('\n\n')[0]).replace('**','')+'</p><p>통과는 정답 검수가 끝난 정확도를 뜻하지 않습니다. offline 평가이며 실시간 재생 검증은 별도입니다.</p><p><a href="REPORT.md">상세 보고서</a> · <a href="../../output/dataset1_target_20260929_run01/comparison.csv">전체 CSV</a></p><table><tr><th>번호 / 영상</th><th>전체 번호 관측</th><th>판정</th><th>확인</th></tr>'
for x in rows:
 body+=f'<tr><td>{html.escape(x["target"])}<br>{html.escape(Path(x["video_path"]).name)}</td><td>{"있음" if x["raw_exact"] else "없음"}</td><td>{"통과" if x["matched"] else "보류"}</td><td><a href="../../{html.escape(x["video_path"])}">원본</a> · <a href="../../output/dataset1_target_20260929_run01/{x["folder"]}/annotated.mp4">판정 영상</a></td></tr>'
(work/'review.html').write_text(page('dataset1 전체 79개 결과',body+'</table>'))
notes={
18:('반복 근거 부족','측면 07은 7.2·7.4초에 정확히 읽음(0.989·0.998). 2회여서 3회 반복 조건 미충족.'),
23:('LED 충돌 + 반복 부족','측면 13은 4.8·5.0초에 정확히 읽음. 앞서 LED에서 읽은 11이 최근 충돌로 남고, 정확한 관측도 2회뿐.'),
24:('버스 검출 + 작은 번호 OCR','33개 샘플 중 25개에서 버스 미검출. 초반 작은 목표 버스가 보임. 0초 수동 번호 crop도 1993으로 오독하여 검출 개선만으로 해결된다고 단정할 수 없음.'),
26:('후보 제한 + 반복 부족','660을 0.4·1.0초에 0.998·0.997로 읽음. incomplete_scan_or_track_jump로 근거가 초기화됨. 해당 프레임 후보 개수 제한도 발생. 정확한 관측 2회.'),
27:('경계 필터 + 점수 + 충돌','6712를 읽지만 1.8·2.8초는 경계 필터로 제외. 2.0·2.2초는 통과 가능한 점수, 2.4초는 0.891. 2.6초의 6717 오독도 충돌에 관여.'),
28:('번호 영역 분리','자동 영역은 7011 중 70만 포함. 2.0초 전체 번호 수동 crop은 같은 PARSeq로 7011, 0.987.'),
29:('측면 영역 분리 + OCR 불확실','자동 영역은 M을 빠뜨리고 1131/1731로 오독. 전체 측면 crop은 M7731을 읽지만 최소 점수 0.666으로 현재 신뢰도 기준 미달.'),
34:('목표 버스 대응 미확정','선택한 3.0·5.2초의 큰 녹색 버스는 영상에서도 6716으로 보이고 OCR도 6716. 이를 단순히 6717 오독으로 분류할 수 없음. 영상 내 목표 6717의 위치는 이번 확인으로 확정하지 않았고 라벨도 변경하지 않음.'),
40:('M과 숫자 분리 + 측면 필터','M과 6628이 따로 검출됨. 측면 6628은 폭 필터로도 제외. 전체 M6628 수동 crop은 0.988로 정확히 읽음.'),
41:('M과 숫자 분리 + 측면 필터','앞 영상과 같은 6초 측면 crop에서 전체 M6628을 0.988로 읽음. 유사 영상이므로 독립된 다양한 환경 2개로 해석하면 안 됨.'),
43:('LED 손상 + 반복 부족','105는 5.8초 0.778, 6.0초 0.931. 일부 프레임에서 읽지만 높은 점수의 반복 근거 부족.'),
46:('측면 필터 + 시간 간격','3500을 여러 번 읽음. 6.6초 측면 PRINT는 0.997인데 좁은 폭 필터로 제외. 나머지 높은 점수 관측은 1.2초 창 내 3회 조건을 못 채움.'),
50:('표기 별칭 불일치','차량에 030이라고 표기. OCR도 030을 반복해서 읽지만 목표 30과 엄격 비교해 불일치. 노선별 검증된 표기 별칭이 필요.'),
54:('표기 별칭 불일치','PRINT/LED의 072를 읽음. 목표 72와 표기가 달라 불일치. 모든 번호의 0을 일괄 제거해서는 안 됨.'),
55:('표기 별칭 불일치','072를 19개 후보에서 읽음. 목표 72와의 표기 차이가 핵심.'),
57:('가림·LED 품질 + 반복 부족','799를 0.2·0.8·1.0·1.2초에 읽지만 높은 점수는 0.2·1.2초 두 번. 이후 다른 버스가 가리고 LED 번짐이 있어 반복 조건 미충족.'),
58:('표기 별칭 불일치','전면 PRINT 085를 30개 후보에서 읽음. 목표 85와의 표기 차이가 핵심.'),
59:('표기 별칭 불일치','미니버스도 28개 샘플 중 14개에서 검출됨. 선명한 전면 PRINT 085를 9개 후보에서 읽음. 검출 전무가 아니라 목표 85와 표기 차이.'),
62:('번호 영역 분리 + 경계 필터','1.2초 자동 영역은 60만 포함. 전체 602 수동 crop은 0.913으로 읽음. 0.8초 602는 0.770이며 경계 필터도 적용.')}
dout=R/'output/diagnosis_20260929_run01';dw=R/'workbench/diagnosis_20260929_run01';dw.mkdir(exist_ok=True)
records=json.loads((dout/'inputs.json').read_text());probes={p['folder']:p for p in json.loads((dout/'manual_crop_probe.json').read_text())};diag=[]
intro='dataset2 정차 보류 19개를 사용자 메모, 프레임, OCR·판정 로그와 대조했습니다. 추가 육안 분석을 사용자에게 요청하는 단계는 끝났습니다. 5개는 30/030·72/072·85/085 표기 불일치이고, 번호 영역 분리와 올바른 측면 인식의 필터 제외도 확인됐습니다. 모델 교체 전에 영역 검출·표기 매핑·필터를 보완할 근거가 있습니다.'
body='<p>'+intro+'</p><p>6개 번호 영역을 직접 지정해 같은 PARSeq로 시험: 5개 문자열 일치, 그중 4개 최소 토큰 점수 ≥0.9. 수동 crop 진단이며 자동 영상 인식률 개선 결과가 아닙니다. M6628 두 영상은 유사 장면입니다.</p><p><a href="REPORT.md">상세 보고서</a> · <a href="../../output/diagnosis_20260929_run01/diagnosis.csv">19개 CSV</a></p>'
md='# dataset2 정차 보류 19개 원인 확인\n\n'+intro+'\n\n원인별 대책: 전체 번호 영역을 유지하는 검출/영역 병합, 노선별 검증된 표기 별칭, 측면 폭 필터와 후보 제한 처리 개선을 우선합니다. 반복 기준을 낮추거나 빈 숫자를 채우는 것은 이번에 시행하지 않았습니다. 수동 crop 성공은 자동 처리 성능과 다릅니다. 273은 선택한 작은 crop에서도 실패했고 M7731은 읽었지만 점수 미달이므로 OCR/영상 품질 한계도 남습니다. 6717 영상의 선택 프레임에 보인 6716은 별도 버스 대응 확인이 필요하며 라벨을 임의 변경하지 않았습니다.\n\n활성 dataset2는 미정차 6개 제외 후 64개, 기존 run02 통과 45개(70.3%), 보류 19개입니다. 이번에는 64개 전체를 재추론하지 않았습니다. 성공 45개의 정차 분류가 없으므로 정차 버스 전체 성공률은 계산할 수 없습니다.\n\n'
for r in records:
 n=int(r['folder'].split('_')[1]);cause,note=notes[n];r.update(cause=cause,analysis=note);diag.append({k:v for k,v in r.items() if k!='times'})
 md+=f'## {r["video_path"]}\n\n- 사용자 메모: {r["user_note"]}\n- 확인 결과: {note}\n\n'
 body+=f'<details><summary>{html.escape(Path(r["video_path"]).name)} — {cause}</summary><p>사용자 메모: {html.escape(r["user_note"])}</p><p>{note}</p><a href="../../{html.escape(r["video_path"])}">원본 영상</a><p>아래 빨간 영역은 당시 OCR 후보이며, 항상 노선 번호를 뜻하지 않습니다.</p><img loading="lazy" src="../../output/diagnosis_20260929_run01/{r["folder"]}_sheet.jpg">'
 if r['folder'] in probes:
  p=probes[r['folder']];body+=f'<p>전체 번호 수동 crop → {p["text"]}, 최소 토큰 점수 {p["min_token_score"]:.3f} (정답 확률 아님)</p><img src="../../output/diagnosis_20260929_run01/{p["crop"]}">'
 body+='</details>'
writecsv(dout/'diagnosis.csv',diag);(dw/'REPORT.md').write_text(md+'\n[19개 근거 이미지와 영상](review.html)\n');(dw/'review.html').write_text(page('dataset2 정차 보류 19개 — 원인 확인 완료',body))
# Preserve review controls but replace stale requests with completed results.
p=R/'workbench/index.html';text=p.read_text();a=text.index('<p>추가 촬영');b=text.index('<main>',a)
text=text[:a]+'''<section><strong>2026-09-29 실행 완료</strong><p>dataset1 전체 79개: 번호 관측 70개(88.6%), 판정 통과 48개(60.8%). dataset2 정차 보류 19개: 원인 확인과 수동 번호 crop OCR 실험 완료.</p><p><a href="dataset1_target_20260929_run01/review.html">dataset1 전체 결과 79개</a> · <a href="diagnosis_20260929_run01/review.html">dataset2 보류 19개 원인과 증거</a></p><p>추가 촬영이나 같은 영상의 재분류를 지금 요청하지 않습니다. 다음 우선 작업은 번호 영역 검출, 노선별 표기 별칭, 측면 필터 보완입니다. 아래는 기존 확인 기록을 열거나 수정하는 도구입니다.</p></section>
'''+text[b:]
a=text.index('<section><strong>그다음은');b=text.index('</section>',a)+len('</section>');text=text[:a]+text[b:]
text=text.replace('먼저 볼 보류 영상 19개','기존 보류 영상 19개').replace('전체 64개 — 정차별 성공률 집계용','활성 영상 전체 64개').replace('나중에 볼 성공 영상 45개','기존 통과 영상 45개')
# Seed imported classifications without overwriting browser edits.
seed={x['video_path']:{'stop_status':x['stop_status'],'note':x['note']} for x in read(R/'workbench/dataset2_stop_review.csv') if x['stop_status']}
marker='// 사용자 확인: 660'
pos=text.index(marker);text=text[:pos]+'const importedReview='+json.dumps(seed,ensure_ascii=False)+';\nfor(const [path,value] of Object.entries(importedReview)){if(!saved[path])saved[path]=value;}\n'+text[pos:];p.write_text(text)
for p in dout.glob('manual_preview_*.png'):p.unlink()
print('Reports complete:',raw,matched,'diagnosed',len(diag))
