# 파일명: scripts/extract_crops.py
# 설명: data/video/의 원본 영상에서 지정한 시각과 좌표로 번호 이미지를
#       추출하고 data/crops/에 저장하는 실행 파일.

import argparse
import csv
import io
import math
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg
from PIL import Image
from bus_route.video import color_filters


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "sample_id", "video_id", "event_id", "timestamp_s", "split", "route_id_gt",
    "visible_text_gt", "x1", "y1", "x2", "y2", "visibility", "display_type",
    "gt_source", "crop_path", "color_mode",
]


def extract_frame(video: Path, timestamp: float, color: str) -> tuple[Image.Image, str]:
    """FFmpeg로 회전 정보를 적용한다. 좌표는 회전 후 전체 프레임 기준."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filters, mode = color_filters(video, color)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-ss", str(timestamp), "-map", "0:v:0", "-vf", filters,
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode or not result.stdout:
        raise RuntimeError(
            "프레임 추출 실패: " + result.stderr.decode(errors="replace")
            if result.returncode else "해당 시각에 프레임이 없습니다. 영상 길이를 확인하세요."
        )
    with Image.open(io.BytesIO(result.stdout)) as image:
        return image.convert("RGB"), mode


def main() -> None:
    parser = argparse.ArgumentParser(description="영상 프레임 확인 및 수동 번호 crop 추출")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True, help="영상 재생 시각(초)")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--sample-id", help="crop 파일명으로 쓸 영문/숫자/밑줄/하이픈 ID")
    parser.add_argument("--text", default="", help="보이는 정답. 03처럼 선행 0 유지")
    parser.add_argument("--route", default="", help="전체 노선 정답")
    parser.add_argument("--event-id", default="")
    parser.add_argument("--split", choices=["train", "val", "test", "dev"], default="dev")
    parser.add_argument("--visibility", choices=["readable", "partial", "no_route", "occluded", "unverified"], default="unverified")
    parser.add_argument("--display-type", choices=["LED", "print"], default="LED")
    parser.add_argument("--gt-source", default="")
    parser.add_argument("--color", choices=["auto", "sdr", "hdr"], default="auto")
    parser.add_argument("--crops-dir", type=Path, default=ROOT / "data/crops")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/annotations.csv")
    parser.add_argument("--frame-output", type=Path, help="crop 전 전체 프레임 저장 경로")
    args = parser.parse_args()
    if not args.video.is_file() or not math.isfinite(args.time) or args.time < 0:
        parser.error("존재하는 영상과 0 이상의 유한한 시각을 지정하세요.")
    if args.bbox is None and args.frame_output is None:
        parser.error("--frame-output 또는 --bbox를 지정하세요.")
    rows = []
    if args.bbox is not None:
        if not args.sample_id or not re.fullmatch(r"[A-Za-z0-9_-]+", args.sample_id):
            parser.error("--bbox 사용 시 영문·숫자·_·- 로 된 --sample-id가 필요합니다.")
        if not args.event_id:
            parser.error("--bbox 사용 시 같은 버스의 접근을 식별할 --event-id가 필요합니다.")
        if args.annotations.exists():
            with args.annotations.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(line for line in stream if not line.startswith("#"))
                if reader.fieldnames and reader.fieldnames != FIELDS:
                    parser.error("기존 annotations 헤더가 지원 형식과 다릅니다. 다른 저장 경로를 지정하세요.")
                rows = list(reader)
        if any(r["sample_id"] == args.sample_id for r in rows):
            parser.error("sample-id가 기존 데이터와 중복됩니다.")
        if any(r["video_id"] == args.video.name and r["event_id"] == args.event_id
               and r["split"] != args.split for r in rows):
            parser.error("같은 영상·event-id에는 같은 split을 지정하세요.")
        crop_path = args.crops_dir / f"{args.sample_id}.png"
        if crop_path.exists():
            parser.error(f"기존 crop을 덮어쓰지 않습니다: {crop_path}")
    if args.frame_output and args.frame_output.exists():
        parser.error(f"기존 프레임을 덮어쓰지 않습니다: {args.frame_output}")
    image, color_mode = extract_frame(args.video.resolve(), args.time, args.color)
    if args.bbox is not None:
        x1, y1, x2, y2 = args.bbox
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            parser.error(f"bbox가 이미지 범위를 벗어납니다. 회전 후 크기: {image.size}")
    if args.frame_output:
        args.frame_output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.frame_output)
        print(f"프레임: {args.frame_output} / size={image.size} / color={color_mode}")
    if args.bbox is None:
        return
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(args.bbox)).save(crop_path)
    row = dict(zip(FIELDS, [
        args.sample_id, args.video.name, args.event_id, str(args.time), args.split,
        args.route, args.text, *map(str, args.bbox), args.visibility, args.display_type,
        args.gt_source, str(crop_path.resolve().relative_to(ROOT))
        if crop_path.resolve().is_relative_to(ROOT) else str(crop_path.resolve()), color_mode,
    ]))
    args.annotations.parent.mkdir(parents=True, exist_ok=True)
    with args.annotations.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# 파일명: annotations.csv\n# 설명: 번호 crop의 원본·좌표·정답. 좌표는 회전 후 전체 프레임 기준.\n")
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([*rows, row])
    print(f"crop: {crop_path} / annotations: {args.annotations}")


if __name__ == "__main__":
    main()
