# gildongmu-bus-route

버스 영상에서 YOLO11n으로 버스를 추적하고 CRAFT와 PARSeq로 번호를 읽는 실험입니다.
직접 지정한 crop 평가와 자동 영상 추론을 모두 지원합니다.

원본 영상과 crop을 포함한 로컬 검토 페이지는 `workbench/README.md`에 있습니다.

## 구성

- `core/bus_route/recognizer.py`: 모델 로딩, 공식 전처리, 추론.
- `core/bus_route/pipeline.py`: 통합앱과 같은 프레임 단위 버스 번호 인식 경로.
  버스 검출 → 번호 표시 영역 검출 → PARSeq → 목표 노선 판정 순서로 실행합니다.
  - `evidence.py`: 후보 선택, 번호 영역 복구, 중복 버스 후보 복구 규칙(영상 평가 실행기와 공유).
  - `observed.py`: 목표가 아닌 다른 버스 번호의 반복 확인.
  - `continuity.py`: 검출기 ID가 사라지거나 바뀌어도 같은 차량으로 이어 붙이기.
  - `led_diagnostics.py`: LED 표시기 주사 누락 진단(기록 전용, 판정에 영향 없음).
  - `models.py`, `base.py`: 버스·번호 영역 검출기와 공통 입출력 형식.
- `scripts/extract_crops.py`: 원본 영상 → 프레임/crop → 정답 CSV 기록.
- `scripts/evaluate.py`: 단일 이미지 또는 CSV 목록 추론·평가.
- `data/video/`, `data/crops/`, `data/annotations.csv`: 원본, crop, 라벨.
- `output/`: 실행별 예측 CSV, 요약 JSON, 설치 패키지 목록.
- `workbench/`: 분석 보고서와 코드 변경 내역.
- `.cache/torch/`: 공식 PARSeq 코드와 가중치 캐시(Git 제외).

## 설치

평가 환경은 통합앱(gildongmu-app)의 실행 환경과 같게 유지합니다.
Python 3.12.15, torch 2.14.0+cu130, ultralytics 8.4.152 등 앱의 `requirements.txt`와
같은 버전입니다. 버전이 다르면 같은 코드라도 OCR·검출 결과가 달라집니다.

```bash
UV_PYTHON_INSTALL_DIR=$PWD/.cache/python-runtime uv venv --python 3.12.15 --seed .venv
uv pip install --python .venv/bin/python torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r ../gildongmu-app/requirements.txt
uv pip install --python .venv/bin/python -e .
```

앱의 의존성 버전이 바뀌면 `pyproject.toml`도 같은 버전으로 바꾸고 이 환경을 다시 만듭니다.

모델은 [PARSeq 공식 저장소](https://github.com/baudm/parseq)의
`1902db043c029a7e03a3818c616c06600af574be` commit으로 고정했습니다.
첫 추론은 인터넷 연결이 필요하며 코드와 `parseq-bb5792a6.pt`를 자동 다운로드합니다.
이후 동일 캐시로 실행합니다. 저장소 자체는 `core/`에 복사하지 않습니다.

## 목표 번호 영상 인식

```bash
.venv/bin/python scripts/infer_target_video.py \
  --videos-csv data/dataset2_active.csv --target-from-manifest \
  --output output/target_new_run --device cuda --realtime
```

`--realtime`은 영상을 원래 속도로 공급하며 밀린 프레임 대신 최신 프레임을 처리합니다.
기본 샘플링은 영상 전체의 5fps입니다. CSV의 `target_route`는 사용자가 기다리는 번호를
제공하는 실험용 입력이며 실제 도착 API 연결은 아닙니다.

기본 판정은 전체 번호를 한 번 읽고 모든 문자·EOS 점수가 0.98 이상이면
`recognized_single`, 기존 3회 반복 조건까지 충족하면 `matched_candidate`입니다.
두 상태는 구분해 기록하며 충돌·잘림·번호판 의심·불완전 탐색·지연 검사도 적용합니다.
`--repeat-only`는 기존 반복 조건만 사용하고, `--single-score`는 단일 관측 기준을 지정합니다.
`first_recognized_s`는 두 상태를 포함한 첫 인식, 기존 `first_matched_candidate_s`는
반복 조건을 통과한 첫 시각입니다. API의 `TargetMatcher`는 `single_score`를 명시해야
단일 관측을 활성화합니다.

번호판 필터는 위치·노란 바탕 비율과 밝기 경계를 함께 검사해 옅거나 그림자 속의
번호판을 제외하면서 어두운 차체 위 밝은 측면 PRINT를 보호합니다. 작은 4자리 차체
숫자는 좌우 문맥에 `4자리-4자리` 전화번호가 있는지도 검사합니다. 문맥은 제외 판단에만
사용하며 노선 번호를 보충하지 않습니다. `predictions.csv`의 `text`는 원시 OCR이고,
`eligible`·`rejection_reason`이 노선 근거 허용 여부입니다. 추가 OCR은 `context_checks`에
분리 기록하며 처리 시간과 프레임 나이에 포함됩니다.

점수는 정답 확률이 아닙니다. 단일 관측 실험에서 번호판·광고·손상 LED의 오인식이
확인됐으므로 단일 관측과 반복 확인을 같은 확정 수준으로 취급하지 않습니다.
이 스크립트는 인식 후보를 평가하며 탑승 확정이나 음성 안내를 실행하지 않습니다.

## 프레임 확인과 crop 만들기

먼저 전체 프레임을 저장해 이미지 뷰어에서 번호 영역의 좌표를 확인합니다.

```bash
.venv/bin/python scripts/extract_crops.py \
  --video 'data/video/20260915_갤럭시S24+_세로_05.mp4' \
  --time 6 --frame-output output/galaxy05_6s.png
```

`--time`은 파일 재생 시각(초)입니다. 회전 메타데이터는 자동 적용하며,
좌표는 **회전 후 전체 프레임의 좌상단 기준 픽셀**입니다. 여백을 자동 제거하지 않습니다.
`--bbox X1 Y1 X2 Y2`에서 오른쪽·아래쪽 경계는 포함하지 않습니다.
HDR 메타데이터(HLG/PQ)가 있으면 Hable tone mapping으로 SDR 변환합니다.
`--color hdr` 또는 `--color sdr`로 명시할 수도 있습니다. 이 변환은 누락된 LED 획을 복원하지 않습니다.

아래는 보고서에 기록된 Galaxy 05의 6초 좌표를 사용하는 예입니다.
추출한 이미지를 확인한 뒤 보이는 정답과 가시성 라벨을 지정하세요.

```bash
.venv/bin/python scripts/extract_crops.py \
  --video 'data/video/20260915_갤럭시S24+_세로_05.mp4' \
  --time 6 --bbox 451 768 755 859 \
  --sample-id galaxy05_6s --event-id galaxy05_bus1 \
  --text 7728 --route 7728 --visibility readable \
  --gt-source '보고서 참조 및 추출 crop 육안 확인'
```

기본 split은 `dev`입니다. 이미 관찰한 기존 영상은 개발용으로 사용합니다.
추가 데이터는 사건별로 split을 먼저 정하고 `--split train|val|test`를 지정합니다.
같은 영상·접근 사건의 split 혼합과 기존 sample-id/crop 덮어쓰기는 거절합니다.
동일 차량이 다른 파일에 등장하는 경우의 분리는 사용자가 관리해야 합니다.

`--text`는 보이는 문자열, `--route`는 실제 전체 노선입니다.
예를 들어 `--text 03 --route 서대문03`처럼 분리하고 선행 0을 유지합니다.
확인하지 못한 정답은 비워 두고 기본 `unverified` 가시성을 사용하세요.
CSV의 `#` 설명 행은 제공한 스크립트에서 제외하여 읽습니다.
상대 `crop_path`는 프로젝트 루트 기준입니다.

## 추론

단일 crop은 정답 없이도 실행할 수 있습니다.

```bash
.venv/bin/python scripts/evaluate.py --image data/crops/galaxy05_6s.png
```

CSV에 모인 crop을 평가하려면 다음을 실행합니다.

```bash
.venv/bin/python scripts/evaluate.py
# 특정 split만 평가
.venv/bin/python scripts/evaluate.py --split dev
# GPU를 사용할 수 없는 환경
.venv/bin/python scripts/evaluate.py --device cpu
```

결과는 `output/recognizer_runs/parseq_<UTC시각>/`에 저장됩니다.
`predictions.csv`에는 정답·원시 예측·토큰 점수·완전 일치·오류·처리시간이,
`summary.json`에는 집계·장치·모델 commit·가중치 SHA-256이 기록됩니다.
`environment.txt`는 실행 당시의 패키지 버전 목록입니다.

정확도 분모는 `visibility=readable`이고 `visible_text_gt`가 있는 crop입니다.
이 표본의 추론 오류도 실패로 포함합니다. 판독 불가·미검수 표본은 예측만 기록합니다.
이 값은 버스 접근 사건 성공률이 아닙니다. 숫자만 추출하거나 API 후보로 교정하지 않습니다.
토큰 점수는 EOS를 포함하며 검증된 정답 확률이 아닙니다.

시간은 warm-up 이후 전처리·GPU 전송·추론·디코딩을 포함합니다.
파일 읽기·모델 다운로드·모델 로딩과 스마트폰 전송·음성 지연은 포함하지 않습니다.
소수 crop에서 얻은 p95는 실시간 성능 보장이 아닙니다.

현재 영상 평가의 분모와 한계는 `workbench/README.md`에서 연결되는 로컬 보고서에 정리했습니다.
정식 분석 결과는 `--output output/dataset1_20260916_run02`처럼 데이터셋·분석일·회차를 지정합니다.
