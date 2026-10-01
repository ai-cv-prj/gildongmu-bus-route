"""Publish reviewed results without replacing any prior evaluation artifacts."""
import json,csv,re,html,hashlib
from pathlib import Path
from evaluate import ROOT,OUT,BASE,load
E=html.escape
WB=ROOT/'workbench/region_recovery_20260930_run01'

def table(headers,rows):
 return '<table><thead><tr>'+''.join('<th>'+E(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def main():
 off,rt=[json.loads((OUT/f'{m}_evaluation.json').read_text()) for m in ['offline','realtime']]
 oldoff,oldrt=[json.loads((BASE/f'{m}_evaluation.json').read_text()) for m in ['offline','realtime']]
 reviews=json.loads((OUT/'evidence/review.json').read_text())
 assert all(r['verdict']!='needs_review' and r['review_note'] for r in reviews),'Complete visual review first'
 assert not off['negative_control']['new_recognized'] and not rt['negative_control']['new_recognized']
 provenance=json.loads((OUT/'provenance.json').read_text())
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in provenance.items())
 WB.mkdir(exist_ok=False,parents=True)
 sections=[]
 sections.append(f'<h1>전체 번호 영역 재확인 · 2026-09-30</h1><p>비교 기준: 번호판 제외 적용 후 offline·realtime 각각 <b>106/143(74.1%)</b>. 원본 전체를 5fps로 다시 추론하고, 같은 GPU에서 원래 속도 재생을 순서대로 수행했습니다. 반례 1개는 143개 분모 밖입니다.</p>')
 sections.append(table(['평가','기존','변경 후','증가','감소'],[[s['mode'],'106 / 143',f"<b>{s['new_recognized']} / 143 ({s['new_recognized']/143:.1%})</b>",len(s['gains']),len(s['losses'])] for s in [off,rt]]))
 for s in [off,rt]:
  sections.append('<p>'+s['mode']+' 증가: '+E(', '.join(r['folder']+' '+r['target'] for r in s['gains']) or '없음')+' / 감소: '+E(', '.join(r['folder']+' '+r['target'] for r in s['losses']) or '없음')+'</p>')
 sections.append('''<h2>적용한 변경</h2><ul><li>원래 OCR이 2~4자리 숫자를 최소 토큰 점수 0.9 이상으로 읽었을 때, 버스당 최대 4개 후보에서 좌우 픽셀을 추가로 읽습니다. 목표 번호·파일명·영상별 좌표는 입력받지 않습니다.</li><li>왼쪽은 문자 높이의 0.5배, 오른쪽은 1.5배를 확장합니다. 기존 버스 검색 영역의 경계를 넘는 crop은 사용하지 않습니다. 왼쪽에서는 지원하는 M/N 접두어만 허용하고, 옆 안내문의 숫자를 연결하지 않습니다.</li><li>기존 숫자가 그대로 보존된 더 긴 노선 형식이어야 하며, 서로 다른 여백의 두 crop이 동일한 전체 문자열을 문자·EOS 모두 0.98 이상으로 읽어야 합니다. 오른쪽 확장의 재확인은 왼쪽도 0.5배, 오른쪽도 0.15배 넓혀 앞부분 잘림을 확인합니다. 같은 프레임의 재확인이므로 독립된 두 관측 또는 반복 3회로 세지 않습니다.</li><li>실제 전체 문자열이 확인되면 원래 조각은 원시 로그에 남기고 노선 근거에서 제외합니다. 접두어가 없는 조각을 별도 노선으로 중복 허용하지 않습니다. 번호판·전화번호·버스 귀속·검색 경계 검사는 확장 영역에도 적용합니다.</li><li>기존 단일 crop의 문자 폭 기준 0.28은 유지합니다. 서로 다른 여백에서 전체 문자열이 재확인된 확장 crop에만 최소 문자당 폭/높이 0.20을 적용해 비스듬한 측면 PRINT를 허용합니다. 미확인 원시 crop의 폭 필터를 일괄 완화하지 않았습니다.</li><li>단일/반복 점수, 경쟁 번호, 트랙, incomplete 및 실시간 300ms 최신성 판정은 유지했습니다. 모델 재학습·교체나 정답에 맞춘 문자열 보충은 없습니다.</li></ul>''')
 sections.append('''<h2>중단한 개발 시도도 보존</h2><p>초기에는 왼쪽 숫자 확장도 허용했습니다. 실제 crop에서 옆 안내문의 6과 노선 721이 6721로 합쳐지고, 별도 차체 숫자 238이 1238로 확대되는 문제가 보여 전체 실행을 중단했습니다. 이 실행은 최종 성능에 포함하지 않습니다. <code>output/region_recovery_20260930_run01/development/</code>에 당시 코드·부분 로그·이미지를 보존했습니다. 이후 왼쪽 숫자 연결을 금지하고 오른쪽 확장의 반대편 여백 재확인을 추가한 코드로 전체 평가를 처음부터 다시 실행했습니다.</p>''')
 metrics=[]
 for title,field,key in [('처리 p95 ms','pipeline_ms','p95'),('처리 최대 ms','pipeline_ms','max'),('프레임 나이 p95 ms','frame_age_ms','p95'),('프레임 나이 최대 ms','frame_age_ms','max')]:
  a,b=oldrt['timing'][field][key],rt['timing'][field][key];metrics.append([title,f'{a:.2f}',f'{b:.2f}',f'{b-a:+.2f}'])
 metrics.extend([['처리 프레임',oldrt['processed_frames'],rt['processed_frames'],rt['processed_frames']-oldrt['processed_frames']],['샘플 폐기',oldrt['dropped_samples'],rt['counts']['dropped_samples'],rt['counts']['dropped_samples']-oldrt['dropped_samples']],['프레임 나이 300ms 초과',oldrt['frame_age_over_300ms_samples'],rt['counts']['age_over_300ms'],rt['counts']['age_over_300ms']-oldrt['frame_age_over_300ms_samples']]])
 sections.append('<h2>실시간 지연</h2>'+table(['지표','기존','변경 후','차이'],metrics))
 sections.append(f"<p>추가 확장 OCR {rt['counts'].get('expansion_ocr_crops',0):,}개 / 확인 OCR {rt['counts'].get('confirmation_ocr_crops',0):,}개. RTX 5080 Laptop GPU, 원본 파일의 원래 재생 속도, 최신 프레임 1개 큐, 5fps. offline과 realtime은 GPU에서 동시에 실행하지 않았습니다. 처리 시간은 검출·OCR·판정까지, 프레임 나이는 원래 재생 시각에서 판정 종료까지입니다. 휴대폰 전송·음성 안내는 포함하지 않습니다. 기존 실행과의 차이는 추가 OCR 비용과 실행 환경 변동을 함께 포함합니다.</p>")
 summaries=json.loads((OUT/'realtime/summary.json').read_text())
 slow=sorted(enumerate(summaries[:143],1),key=lambda x:x[1]['pipeline_ms']['p95'],reverse=True)[:5]
 sections.append(table(['느린 영상','처리 p50 / p95 / 최대 ms','나이 p95 / 최대 ms','폐기'],[[f'video_{i:02} · '+E(Path(s['video']).name),' / '.join(f"{s['pipeline_ms'][k]:.1f}" for k in ['p50','p95','max']),' / '.join(f"{s['replay_frame_age_ms'][k]:.1f}" for k in ['p95','max']),s['dropped_samples']] for i,s in slow]))
 sections.append('<h2>번호판·광고 제외와 정상 측면 보호</h2>'+table(['모드','영상','시험 번호','원시 OCR / 적격 관측','해당 번호를 목표로 통과','사유'],[[s['mode'],r['folder'],r['target'],f"{r['raw']} / {r['eligible']}",r['recognized'],E(str(r['reasons']))] for s in [off,rt] for r in s['reviewed_diagnostics'] if r['id'] in [19,24,25]]))
 sections.append('<p><a href="../plate_filter_20260929_run01/review.html">보존한 번호판·광고 원본 이미지 및 측면 PRINT·LED 보호 이미지</a>. 아래 표는 이번 원본 재추론에서 같은 시각의 관측을 확인한 결과입니다.</p>')
 sections.append(table(['모드','영상','번호','근거 시각','원시 / 적격','영상 인식'],[[s['mode'],r['folder'],r['target'],r['anchor_s'],f"{r['raw']} / {r['eligible']}",r['recognized']] for s in [off,rt] for r in s['protected_anchors']]))
 sections.append('<p>보호 표의 03은 기존에도 경쟁 번호 충돌로 영상 판정이 보류된 사례입니다. 정상 PRINT 관측 자체가 이번에도 적격으로 유지됐는지와 영상 최종 인식을 구분합니다.</p>')
 sections.append('<p>전체 기존 적격 목표 관측을 시각·문자·좌표까지 대조하면 offline 손실은 0개입니다. realtime에서는 16개가 동일 좌표로 재현되지 않았습니다. 11개는 같은 시각의 전체 목표 문자열이 다른 좌표로 유지됐고, 4개는 샘플 폐기, 1개는 video_11의 3.2초 271→71 부분 영역 변화입니다. 지정 보호 anchor 16개는 모두 적격으로 유지됐지만 모든 관측이 동일하다는 뜻은 아닙니다.</p>')
 sections.append('<p>7017 원본 영상에 목표 7016 지정: offline·realtime 모두 미인식. 전체 143개 및 반례의 프레임별 판정·원시 OCR·확장 OCR·재확인 OCR을 보존했습니다. 회귀 검사 44개 통과.</p>')
 sections.append('<h2>새 오인식과 다른 목표 검사</h2>')
 for s in [off,rt]:
  new=[r for r in reviews if r['mode']==s['mode'] and r['kind']=='new_alternative']
  errors=[r for r in new if r['verdict']=='false_recognition']
  sections.append(f"<p>{s['mode']}: 다른 목표 {s['counts']['alternative_cases']:,}조합 중 {s['alternative_passed']}조합 통과. 동일 모드의 기존 로그보다 새 통과 {len(new)}조합, 육안 확인된 새 오류 {len(errors)}조합. 실제 다른 버스와 별칭 중복을 포함하므로 전체 통과 수를 오인식 건수로 해석하지 않습니다.</p>")
 sections.append('<p><b>실시간에서 재등장한 광고 연도 2026:</b> 기존 offline에서도 통과했던 원시 OCR입니다. 이전 실시간은 1.0초 프레임을 폐기했고, 이번 실행은 앞선 0.6초를 폐기한 뒤 1.0초를 처리해 2026이 통과했습니다. 해당 프레임 나이는 약 190.30ms였고 확장 OCR 출처가 아닙니다. 샘플 폐기에 기대어 차단됐던 기존 오류의 재등장이며, 번호판·전화번호 필터가 광고 연도까지 해결한 것은 아닙니다. 기존 realtime 대비 새 통과/오류 1조합으로 집계합니다.</p>')
 sections.append('<p><b>나머지 실시간 새 오류:</b> 150은 유리창에 비친 간판으로 보이는 작은 문자 영역입니다. 기존 offline에서도 통과했지만 이전 realtime은 나이 422.09ms여서 보류했고, 이번에는 218.27ms로 통과했습니다. 71은 실제 측면 271의 잘린 부분입니다. 이번 실행에서 샘플 폐기 이력과 트랙/검색 박스가 달라져 3.2초 CRAFT 영역이 일부만 잡혔고, 289.94ms로 통과했습니다. 세 새 통과 모두 확장 OCR 출력은 아니지만, 실행 시간과 추적의 상호작용까지 포함한 이번 realtime 결과의 오류로 공개합니다. 차체 식별번호 일부 238이 이번에 미인식인 것도 나이 340.16ms로 보류된 결과이며, 새 필터 효과가 아닙니다.</p>')
 sections.append('<p>대안 목표는 기존·현재 OCR 문자열의 합집합, 활성 목표 목록, 목표 각 숫자의 ±1 변형으로 구성했습니다. 각 대안은 기존·현재 로그에서 동일한 판정기로 재검사했습니다. 새 통과는 아래 원본 위치와 crop으로 검토했으며, 모든 프레임의 모든 출력에 정답 라벨을 단 전수 정확도가 아닙니다.</p>')
 sections.append(table(['모드','기존 검토 영상','시험 번호','육안 분류','기존 → 현재 통과'],[[s['mode'],r['folder'],r['target'],r['verdict'],f"{r['old_recognized']} → {r['recognized']}"] for s in [off,rt] for r in s['reviewed_diagnostics']]))
 sections.append('<h2>원본 이미지 근거</h2><p>주황: 버스 / 빨강: 실제 읽은 전체 crop / 청록: 확장 전 조각. gain은 새 목표 인식, new_alternative는 새로 통과한 다른 목표, removed_alternative는 기존에 통과했지만 이번에는 통과하지 않은 다른 목표입니다.</p>')
 cards=[]
 for r in reviews:
  prefix=f'../../output/{OUT.name}/evidence/{r["id"]:03}'
  source=f'<p>확장 전 {E(r["recovered_from"]["text"])}<br><img class="crop" src="{prefix}_source.png"></p>' if r['recovered_from'] else ''
  cards.append(f'''<article><h3>{r['id']} · {r['mode']} · {r['kind']} · {r['folder']} · {E(r['target'])}</h3><p>{E(r['review_note'])}</p><p>{r['first_s']}초 · 원시 {E(r['ocr_text'])} · 최소 점수 {r['score']:.4f} · {E(r['verdict'])}</p><a href="{prefix}_context.jpg"><img class="context" loading="lazy" src="{prefix}_detail.jpg"></a>{source}<p>전체 crop<br><img class="crop" loading="lazy" src="{prefix}_crop.png"></p><a href="../../{E(r['video_path'])}#t={r['first_s']}">원본 영상</a></article>''')
 sections.append('<div class="grid">'+''.join(cards)+'</div>')
 rows=[list(csv.DictReader((OUT/f'{m}_comparison.csv').open(encoding='utf-8-sig'))) for m in ['offline','realtime']]
 sections.append('<h2>143개 전체와 반례</h2>'+table(['영상','목표','offline 기존 → 현재','realtime 기존 → 현재','첫 인식 offline / realtime','영상'],[[E(a['folder']+' · '+Path(a['video_path']).name),E(a['target']),a['old_recognized']+' → '+a['new_recognized'],b['old_recognized']+' → '+b['new_recognized'],a['first_s']+' / '+b['first_s'],f'<a href="../../{E(a["video_path"])}">원본</a> · <a href="../../output/{OUT.name}/realtime/{a["folder"]}/annotated.mp4">처리 결과</a>'] for a,b in zip(*rows)]))
 sections.append('''<h2>해석의 범위와 남은 과제</h2><p>이 수치는 목표 번호가 있는 개발용 영상에서 목표가 한 번 이상 인식된 비율입니다. 모든 출력의 정답률·탑승 안전도·독립 일반화 성능과 다릅니다. 같은 장면이나 같은 차량을 담은 유사 영상도 각각 셉니다. 반사가 심해 원래 숫자도 낮은 점수인 M7731은 이번 방식으로 해결되지 않았습니다. 검색 경계 밖으로 잘린 번호, 미검출 버스, 강한 기울기·빛번짐·손상 LED, 기존의 한글·광고 오독도 남습니다. 서로 다른 여백의 재확인도 동일한 모델·픽셀을 쓰므로 같은 오독에 동의할 수 있습니다.</p><p>영문 접두어를 추정해 붙이거나 정답 숫자에 맞춰 crop을 고르지 않습니다. 제한한 왼쪽 숫자 누락은 이번에 복구하지 않으며, 향후 실제 문자 영역과 줄 관계를 더 정확히 검출해야 합니다.</p>''')
 sections.append(f'<p>재현 스크립트: <code>scripts/experiments/{OUT.name}/</code>. 로그·입력·설정·코드 해시·전체 비교 CSV: <a href="../../output/{OUT.name}/offline_evaluation.json">offline 집계</a> · <a href="../../output/{OUT.name}/realtime_evaluation.json">realtime 집계</a> · <a href="../../output/{OUT.name}/evidence/review.json">이미지 판독 기록</a>. 원본·정답 CSV·기존 평가 결과는 덮어쓰지 않았습니다.</p>')
 doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>전체 번호 영역 개선 검증</title><style>body{font:16px system-ui;line-height:1.7;max-width:1200px;margin:30px auto;padding:0 18px;color:#182d3a;background:#f5f8fa}h1,h2{line-height:1.3}h2{margin-top:45px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:9px;border:1px solid #cdd8df;text-align:left;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}article{background:white;padding:16px;border:1px solid #cdd8df;border-radius:8px}.context{width:100%;height:300px;object-fit:contain}.crop{max-width:100%;max-height:150px;min-height:40px}a{color:#125fa5}code{overflow-wrap:anywhere}</style><a href="../index.html">← 결과 시작점</a>'''+''.join(sections)+'</html>'
 (WB/'review.html').write_text(doc)
 index=ROOT/'workbench/index.html';before=index.read_text();(WB/'previous_index.html').write_text(before)
 current=f'<section id="current-results"><h2>전체 번호 영역 개선 · 2026-09-30</h2><p>기존 106/143 → offline <b>{off["new_recognized"]}/143 ({off["new_recognized"]/143:.1%})</b>, realtime <b>{rt["new_recognized"]}/143 ({rt["new_recognized"]/143:.1%})</b>.</p><p><a href="{OUT.name}/review.html"><b>정확도·새 오인식·지연 및 원본 이미지 근거</b></a></p><p>7017→7016 반례 미인식 유지. 원본과 기존 결과는 보존했습니다.</p></section>'
 updated,n=re.subn(r'<section id="current-results">.*?</section>',current,before,count=1,flags=re.S);assert n==1
 if 'plate_filter_20260929_run01/review.html' not in updated:
  updated=updated.replace('<summary>보존한 분석·이전 기록</summary><ul>','<summary>보존한 분석·이전 기록</summary><ul><li><a href="plate_filter_20260929_run01/review.html">이전 번호판 제외 검증 106/143</a></li>')
 updated=updated.replace('결과 중심 보고서 열기 — 모델·74.1% 인식률·143개 이미지·실패 원인·다음 방향','이전 팀 보고서 열기 — 2026-09-29 기준 74.1%·143개 이미지·실패 원인')
 index.write_text(updated)
 print(WB/'review.html')
if __name__=='__main__':main()
