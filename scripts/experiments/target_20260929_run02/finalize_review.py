"""실제 crop을 보고 내린 검토 결과와 보류 원인을 최종 보고서에 기록한다."""
import csv,json,html,hashlib,shutil
from pathlib import Path
from bus_route.detector import ROOT
from bus_route.target import exact_route
r=ROOT/'output/target_20260929_run02';work=ROOT/'workbench/target_20260929_run02'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
summary=json.loads((r/'comparison_summary.json').read_text());rows=read(r/'comparison.csv');flags=json.loads((r/'alternative_audit/evidence.json').read_text())
verdicts={
 ('video_22','14'):('confirmed_false','전면 광고 한글 ‘고장’을 14로 반복 오독했다. 2.6·2.8·3.6초에 점수 기준을 충족해 목표 14의 판단 조건을 통과한다.'),
 ('video_33','7611'):('other_bus_visible','파일명 목표 602와 별도로 화면 오른쪽 다른 버스의 측면 PRINT 7611이 보인다.'),
 ('video_34','6716'):('other_bus_visible','실제 화면의 해당 트랙 전면 LED는 6716이다. 파일명 6717과 다르다는 이유만으로 오독이라고 판단하면 안 된다.'),
 ('video_36','270'):('other_bus_visible','화면 오른쪽 다른 버스의 후면 PRINT 270이다.'),
 ('video_38','5714'):('other_bus_visible','화면 오른쪽 다른 버스의 측면 PRINT 5714이다.'),
 ('video_38','6712'):('other_bus_visible','화면 오른쪽 다른 버스의 후면 PRINT 6712이다.'),
 ('video_68','7101'):('needs_review','앞쪽 7713 버스와 다른 뒤쪽 트랙의 표기를 7101로 읽었다. 작은 crop이라 전체 노선의 독립 육안 확정은 보류한다.'),
}
for f in flags:
 f['verdict'],f['review_note']=verdicts.get((f['folder'],f['target_route']),('needs_review','추가 육안 검토 필요'))
counts={x:sum(f['verdict']==x for f in flags) for x in ['confirmed_false','other_bus_visible','needs_review']}
summary['alternative_visual_review']=counts
summary['interpretation']='영상별 목표 확인 성공률이며 precision 또는 탑승 안전도 아님. 다른 목표 검사에서 확인된 오통과가 남아 있음.'
dump(r/'comparison_summary.json',summary);dump(r/'alternative_audit/reviewed_evidence.json',flags)
failures=[]
for row in rows[:70]:
 if row['new_match']=='True':continue
 preds=read(r/row['folder']/'predictions.csv');valid={};target=[]
 for p in preds:
  if exact_route(p['text'],row['target_route']):
   target.append(p)
   if p['eligible']=='True' and min(json.loads(p['token_scores']))>=.9:
    valid.setdefault(p['track_id'],set()).add(float(p['timestamp_s']))
 most=max(map(len,valid.values()),default=0)
 why='OCR 출력에 목표 문자열 없음 (영상에 번호가 없다는 뜻은 아님)' if not target else '한 트랙에서 유효한 고신뢰 전체 일치 3프레임 미확보' if most<3 else '반복 간격·트랙·경쟁 번호·탐색 완전성 조건으로 보류'
 failures.append({'folder':row['folder'],'video_path':row['video_path'],'target_route':row['target_route'],'reason':why,'high_quality_times_by_track':{k:sorted(v) for k,v in valid.items()},'target_crop_rejections':sorted({p['rejection_reason'] for p in target if p['rejection_reason']})})
dump(r/'hold_reasons.json',failures)
lines=['','## 다른 목표의 실제 crop 검토',f'파일명과 다른 목표에서 통과한 {len(flags)}건의 첫 판단 crop을 검토했다. **명백한 오통과 {counts["confirmed_false"]}건**, 다른 버스의 실제 표기가 보이는 경우 {counts["other_bus_visible"]}건, 독립 육안 판독 보류 {counts["needs_review"]}건이다. 이는 준비한 재생 조합의 사례 수이며 전체 서비스 오확정률이 아니다.',
       '', '**광고 한글을 14로 잘못 확정하는 사례가 남아 있다. 따라서 현재 결과를 사용자에게 탑승 확정으로 안내할 수준으로 평가하지 않는다.** 번호 영역 전용 검출 또는 한글/광고를 구분하는 추가 인식 근거가 필요하다.',
       '', '[다른 목표의 위치·crop 검토](alternative_review.html)', '', '## 왜 일부는 여전히 보류하는가',
       '- 마포07 측면 02: 끝부분 7.2·7.4초의 07은 읽었지만 유효한 두 프레임뿐이라 3프레임 조건 미충족.',
       '- 7016 LED 03: 점수 기준을 통과한 전체 번호는 2.0·2.2초 두 프레임. 번호판 충돌을 제거해도 반복 근거가 부족하다.',
       '- 마포13 불량: 측면 13은 4.8·5.0초 두 프레임. 앞선 LED 오독과 짧은 노출 문제가 남는다.',
       '- 6712: 2.8초의 올바른 OCR crop도 버스 탐색 경계에 닿아 이번 보수 필터에서 제외된다. 결과적으로 고신뢰 유효 근거는 2.0·2.2초 두 프레임이다. 경계 필터의 정상 근거 손실 사례이며, 단순 제외 대신 원본에서 경계를 넓혀 다시 검출하는 후속 개선이 필요하다.',
       '- 660 측면: 정확한 660이 나오는 시점은 있으나 반복 근거와 후보 탐색 완전성 조건을 충족하지 못했다.',
       '', f'전체 보류 {len(failures)}개의 경로·유효 시각·제외 이유는 output/target_20260929_run02/hold_reasons.json에 기록했다.',
       '', '## 첫 판단 영역 검토 범위',
       '통과 45개 영상의 첫 판단 crop과 주변 화면을 확인했다. 이 첫 판단의 위치는 노선 LED/PRINT이며 번호판·광고를 근거로 한 경우는 발견하지 못했다. 이후 모든 출력의 정답을 프레임 단위로 전수 라벨링한 검증은 아니다. 흐린 LED의 개별 글자는 사용자 라벨과 함께 해석해야 한다.',
       '', '## 재실행',
       '```bash',
       '.venv/bin/python scripts/infer_target_video.py --videos-csv output/target_20260929_run02/inputs.csv --target-from-manifest --output output/target_next --device cuda',
       '.venv/bin/python scripts/infer_target_video.py --videos-csv output/target_20260929_run02/replay_inputs.csv --target-from-manifest --output output/target_next/realtime --device cuda --realtime',
       '.venv/bin/python -m unittest discover -s scripts/tests -v',
       '```',
       '', '보고서 생성·검토 재현 스크립트는 scripts/experiments/target_20260929_run02/에 있다. 이 스크립트들은 이번 실행 경로에 고정된 실험 기록이다.']
with (work/'REPORT.md').open('a') as f:f.write('\n'.join(lines)+'\n')
parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>다른 목표 오통과 검토</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px;line-height:1.6}article{border-top:2px solid #aaa;padding:20px 0}.context{max-height:500px;max-width:400px}.crop{max-height:140px;min-height:70px;max-width:350px;object-fit:contain;image-rendering:pixelated}code{overflow-wrap:anywhere}</style><h1>다른 목표의 실제 위치 검토</h1><p>파일명과 다른 번호가 나와도 화면의 다른 버스에 실제로 적힌 번호라면 OCR 오독이 아닙니다. 광고 한글 오독과 독립 판독을 보류한 작은 crop을 구분했습니다.</p><a href="REPORT.md">평가 보고서</a>']
for f in flags:
 base='../../output/target_20260929_run02/alternative_audit/'
 parts.append(f'<article><h2>{f["folder"]} · 목표 {html.escape(f["target_route"])} · {f["first_s"]}초</h2><code>{html.escape(f["video_path"].removeprefix("data/"))}</code><p><b>{f["verdict"]}</b>: {f["review_note"]}</p><a href="{base}{f["context"]}"><img class="context" src="{base}{f["context"]}"></a><img class="crop" src="{base}{f["crop"]}"></article>')
parts.append('</html>');(work/'alternative_review.html').write_text('\n'.join(parts))
p=work/'review.html';s=p.read_text().replace('<h1>',f'<p><strong>다른 목표 검사에서 명백한 오통과 {counts["confirmed_false"]}건이 남았습니다.</strong> <a href="alternative_review.html">오통과·다른 버스 crop 확인</a></p><h1>',1);p.write_text(s)
with (work/'HANDOFF.md').open('a') as f:f.write(f'\n최종 영상 확인 통과 {summary["new_match"]}/70, 이전 38/70 대비 +7개. 다른 목표 검사 {summary["counterfactual_cases"]}조합 중 검토 대상 {len(flags)}건: 명백한 광고 오통과 {counts["confirmed_false"]}, 다른 버스 실제 번호 {counts["other_bus_visible"]}, 육안 보류 {counts["needs_review"]}. 탑승 확정용으로는 부족하다. 노선 영역 분류와 crop 경계 재검출이 다음 과제다. 중간 재실행 폴더는 제거하고 최종 run02만 보존한다.\n')
dump(r/'candidate_audit/visual_review.json',{'reviewed_first_candidate_crops':45,'scope':'첫 판단의 노선 표시 위치 확인; 모든 프레임 문자 정답 전수 검증 아님'})
for name in ['tests','realtime','report']:
 source=Path(f'/tmp/bus_target_run02_{name}.log')
 if source.exists():shutil.copy2(source,r/f'{name}.log')
shutil.copy2('/tmp/bus_target_run02.log',r/'inference.log')
dump(r/'test_results.json',{'command':'.venv/bin/python -m unittest discover -s scripts/tests -v','tests':26,'result':'passed','exit_code':0,'note':'imageio_ffmpeg pipe ResourceWarning 있음; 테스트 실패 없음'})
hashes=json.loads((r/'source_sha256.json').read_text())
for p in Path(__file__).parent.glob('*.py'):hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
dump(r/'source_sha256.json',hashes)
print(json.dumps({'alternative_review':counts,'hold_videos':len(failures)},ensure_ascii=False))
