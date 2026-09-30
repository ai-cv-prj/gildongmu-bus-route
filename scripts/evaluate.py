# 파일명: scripts/evaluate.py
# 설명: 번호 이미지와 정답으로 인식기를 평가하고 예측, 정확도,
#       처리시간을 output/에 저장하는 실행 파일.

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from PIL import Image

from bus_route.recognizer import PARSeqRecognizer


ROOT = Path(__file__).resolve().parents[1]


def load_samples(path: Path, split: str | None) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        required = {"sample_id", "crop_path", "visible_text_gt", "visibility", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV에 다음 열이 필요합니다: {sorted(required)}")
        rows = [row for row in reader if split is None or row["split"] == split]
    if not rows:
        raise ValueError("평가할 표본이 없습니다. crop을 먼저 추출하세요.")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("sample_id가 중복되어 있습니다.")
    for row in rows:
        path = Path(row["crop_path"])
        row["crop_path"] = str(path if path.is_absolute() else ROOT / path)
        if not Path(row["crop_path"]).is_file():
            raise FileNotFoundError(row["crop_path"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="PARSeq 단일 이미지 추론 또는 번호 crop 평가")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path, help="번호 문자열만 자른 이미지")
    inputs.add_argument("--annotations", type=Path, help="기본: data/annotations.csv")
    parser.add_argument("--split", choices=["train", "val", "test", "dev"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, help="새 결과 폴더. 기본은 실행 시각별 폴더")
    args = parser.parse_args()
    if args.image and args.split:
        parser.error("--split은 CSV 평가에서만 사용합니다.")
    if args.image:
        if not args.image.is_file():
            parser.error(f"이미지가 없습니다: {args.image}")
        samples = [{"sample_id": args.image.stem, "crop_path": str(args.image.resolve()),
                    "visible_text_gt": "", "visibility": "unverified", "split": ""}]
    else:
        samples = load_samples(args.annotations or ROOT / "data/annotations.csv", args.split)
    output = args.output or ROOT / "output/recognizer_runs" / datetime.now(timezone.utc).strftime("parseq_%Y%m%dT%H%M%S_%fZ")
    if output.exists():
        parser.error(f"기존 결과를 덮어쓰지 않습니다. 새 --output을 지정하세요: {output}")
    print(f"PARSeq 로딩 및 warm-up ({args.device})…", flush=True)
    recognizer = PARSeqRecognizer(device=args.device)
    output.mkdir(parents=True)
    results = []
    for sample in samples:
        row = dict(sample)
        row.update(text="", token_scores=[], elapsed_ms=None, exact_match=None, error="")
        row["scored"] = sample["visibility"] == "readable" and bool(sample["visible_text_gt"])
        try:
            with Image.open(sample["crop_path"]) as image:
                image.load()  # 파일 디코딩 시간은 추론 측정에서 제외한다.
                row.update(asdict(recognizer.predict(image)))
            if row["scored"]:
                row["exact_match"] = row["text"] == sample["visible_text_gt"]
        except (OSError, ValueError, RuntimeError) as exc:
            row["error"] = str(exc)
            if row["scored"]:
                row["exact_match"] = False
        results.append(row)
        print(f"{sample['sample_id']}: gt={sample['visible_text_gt']!r} → {row['text']!r}"
              f" / {row['elapsed_ms']} ms" + (f" / ERROR: {row['error']}" if row["error"] else ""), flush=True)
    scored = [r for r in results if r["scored"]]
    times = [r["elapsed_ms"] for r in results if r["elapsed_ms"] is not None]
    summary = {
        **recognizer.metadata(), "python": platform.python_version(),
        "command": sys.argv, "sample_count": len(results), "scored_count": len(scored),
        "correct_count": sum(r["exact_match"] is True for r in scored),
        "exact_accuracy": sum(r["exact_match"] is True for r in scored) / len(scored) if scored else None,
        "error_count": sum(bool(r["error"]) for r in results),
        "latency_ms": {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
                       "max": max(times)} if times else None,
        "measurement": "crop preprocessing + transfer + inference + decoding; excludes load/warmup/file IO",
        "evaluation": "readable crops with nonempty visible_text_gt only; NOT bus event accuracy",
    }
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(key for row in results for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "token_scores": json.dumps(row["token_scores"])})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment.txt").write_text(packages.stdout, encoding="utf-8")
    print(f"결과: {output}\n정답 {summary['correct_count']}/{len(scored)} (crop 기준)")
    if summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
