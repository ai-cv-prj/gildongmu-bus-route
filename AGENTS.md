# 프로젝트 작업 지침

- Git 커밋, GitHub 업로드, Linear 정리 작업에서는 [CLAUDE.md](CLAUDE.md)의 커밋 규칙과 `workbench/linear-ticket.md`의 티켓 대장을 확인한다. 티켓 번호가 없는 변경은 대장에 제목·설명을 먼저 준비한다.
- `data/`, `output/`, `workbench/`, `.cache/`는 Git 제외 대상이다. 원본 영상·라벨과 기존 평가 결과를 지우거나 새 결과로 덮어쓰지 않는다.
- 추론 코드 변경에는 `.venv/bin/python -m unittest discover -s scripts/tests -v`를 사용한다. 영상 전체 재평가와 실시간 재생이 필요하면 두 실행을 같은 GPU에서 순서대로 수행한다.
- 작업 내역과 결과물은 `output/`, `workbench/`에만 저장한다. Git 추적 경로에 별도 결과 문서를 만들지 않는다. 사용자용 로컬 결과 진입점은 `workbench/README.md`이다.

- 확정 HTML 보고서는 `workbench/report/YYYYMMDD_주제_report.html`, 재사용 도구는 `workbench/tools/`에 둔다. 같은 날짜·주제는 버전을 붙이고 기존 확정 결과를 덮어쓰지 않는다. 새 `index.html`을 만들지 않는다.
- 일회성 제작·검증 스크립트, 초안, 중간 렌더링과 캐시는 작업 완료 후 삭제한다. 원본·라벨·확정 평가 결과 및 재현에 필요한 고정 소스는 보존한다.
