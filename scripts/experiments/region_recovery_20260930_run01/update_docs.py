"""Prepend the completed result to the local workbench documents."""
import json,hashlib
from pathlib import Path
from evaluate import ROOT,OUT

def main():
 assert (OUT/'artifact_validation.json').exists()
 off,rt=[json.loads((OUT/f'{m}_evaluation.json').read_text()) for m in ['offline','realtime']]
 evidence=json.loads((OUT/'evidence/review.json').read_text())
 errors={m:[r for r in evidence if r['mode']==m and r['kind']=='new_alternative' and r['verdict']=='false_recognition'] for m in ['offline','realtime']}
 timing=rt['timing'];c=rt['counts']
 text=f'''# 2026-09-30 새 세션 시작점 — 전체 번호 영역 개선 완료

최신 결과는 [전체 번호 영역 개선 보고서](region_recovery_20260930_run01/review.html)이며 고정 진입점은 [index.html](index.html)이다. `team_report.html`은 수정 전 106/143 기준의 보존 보고서다. 아래 이전 시작점보다 이 절을 우선한다.

- 원본 활성 143개와 7017→7016 반례 1개를 각각 offline 다음 원래 속도 realtime으로 다시 실행했다. **offline {off['new_recognized']}/143 ({off['new_recognized']/143:.1%}), realtime {rt['new_recognized']}/143 ({rt['new_recognized']/143:.1%})**. 기존은 두 모드 모두 106/143. offline 증가 {len(off['gains'])}·감소 {len(off['losses'])}, realtime 증가 {len(rt['gains'])}·감소 {len(rt['losses'])}.
- 새 인식: `video_25` 6714(2.2초), `video_102` 7011(2.0초), `video_114·115` M6628(6.0초). 뒤의 두 영상은 유사 장면을 각각 센 것이다. 실제 확대 crop으로 전체 번호를 다시 읽은 결과이며 목표 번호·파일명·정답 좌표를 추론 입력으로 사용하지 않는다.
- 새 `core/bus_route/region_recovery.py`와 `scripts/infer_target_video.py`의 `recover_regions`가 기존 숫자 crop 주변을 다시 읽는다. 원시 숫자 ≥0.9, 확장 및 여백 재확인 OCR의 모든 문자·EOS ≥0.98, 같은 전체 문자열, 원래 숫자 보존을 요구한다. 좌측은 M/N 접두어만 허용하며, 우측 확장 확인 때 반대쪽 여백도 넓힌다. 같은 프레임의 두 crop을 반복 관측 2회로 세지 않는다.
- 원래 조각은 로그에 남기되 전체 영역이 확인되면 `superseded_by_full_region`으로 노선 근거에서 제외한다. 등록번호·광고 전화번호·귀속·경계 검사와 300ms 최신성·경쟁 번호·트랙·incomplete 판정은 유지했다. 비스듬한 확장 영역에서 전체 문자열이 재확인된 경우만 문자당 폭/높이 바닥값 0.20(기존 단일 crop 0.28)을 적용한다.
- 처음에는 왼쪽 숫자 연결도 허용했으나 721 옆 안내문의 6을 합친 6721 등 문제가 보여 중단했다. 당시 부분 실행·코드·이미지는 `output/{OUT.name}/development/`에 보존했으며 최종 수치에 포함하지 않는다. 최종 코드는 전체 평가를 처음부터 다시 실행했다.
- 번호판/전화번호 `7284·1577·9350`은 두 모드에서 적격 관측 0개·해당 숫자 목표 인식 0개. 기존 정상 측면 PRINT·LED 보호 anchor 16개 유지. 다만 realtime 전체 관측 좌표 대조에서는 11개 좌표 변경·4개 샘플 폐기·1개 271→71 부분 영역 변화가 있었다. 영상 성공 유지와 모든 관측의 동일성을 구분한다. 7017 원본에 7016을 목표로 지정한 반례는 두 모드 모두 미인식.
- 다른 목표의 새 통과: offline {len(off['new_alternative_flags'])}, realtime {len(rt['new_alternative_flags'])}조합. 이미지 검토상 새 오류는 각각 {len(errors['offline'])}, {len(errors['realtime'])}조합. 기존에 통과하던 부분 번호 70(video_27, 실제 700)은 전체 700 재확인 후 차단됐다. 실시간의 2026은 기존 offline에도 있던 광고 연도 오류가 다시 나타난 것이다. 이전 realtime에서 폐기했던 1.0초 프레임을 이번에는 처리했으며, 확장 OCR 출력은 아니다. 150도 기존 offline 오류가 최신성 조건을 통과한 것이며, 새 71은 샘플 폐기/추적 변화로 271의 일부만 잡힌 원시 OCR이다. 실시간 새 3조합은 모두 오류로 검수했다. 기존의 한글·광고·손상 LED 오독은 여전히 남아 있다.
- realtime {rt['processed_frames']:,}프레임, 처리 p95 {timing['pipeline_ms']['p95']:.2f}ms / 최대 {timing['pipeline_ms']['max']:.2f}ms, 프레임 나이 p95 {timing['frame_age_ms']['p95']:.2f}ms / 최대 {timing['frame_age_ms']['max']:.2f}ms, 폐기 {c['dropped_samples']}개, 나이 300ms 초과 {c['age_over_300ms']}개. 이전 지연과의 비교표·느린 영상은 보고서에 있다. 추가 OCR 비용과 환경 변동이 함께 포함된다.
- 단위·통합 검사 44개 통과. 입력 목록, 코드 해시, 판정 완전성, 보고서 이미지·링크를 검증했다. 원본·사용자 CSV·이전 평가 산출물은 덮어쓰지 않았다. Git 커밋·업로드는 수행하지 않았다.

산출물: `output/{OUT.name}/`의 offline/realtime, evaluation JSON·CSV, `evidence/review.json`, `provenance.json`, `artifact_validation.json`. 재현 스크립트: `scripts/experiments/{OUT.name}/`. 이후 변경은 새 출력 디렉터리를 사용한다.

다음 과제: 심한 반사의 M7731, 검색 경계 밖 잘림, 미검출/추적 누락, 남은 한글·광고·손상 LED 오독. 이번 제한적 좌우 재확인을 범용 전체 노선 검출이나 모든 출력의 정확도 개선으로 해석하지 않는다. 독립된 미사용 영상 검증은 아직 수행하지 않았다.

---

'''
 handoff=ROOT/'workbench/HANDOFF.md'
 before=handoff.read_text();(OUT/'previous_HANDOFF.md').write_text(before)
 handoff.write_text(text+before)
 print('Updated workbench/HANDOFF.md')
if __name__=='__main__':main()
