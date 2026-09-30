"""저장 영상의 YOLO→CRAFT→PARSeq 분석 및 표시 영상/CSV 생성."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch

from bus_route.detector import BusTextDetector, ROOT, CRAFT_COMMIT
from bus_route.recognizer import PARSeqRecognizer
from bus_route.video import sample_frames, video_writer


def save_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict)) else v
                             for k, v in row.items()})


def summarize_tracks(predictions):
    tracks = defaultdict(list)
    for row in predictions:
        if row['track_id'] is not None:
            tracks[row['track_id']].append(row)
    result = []
    for track, rows in tracks.items():
        # 한 프레임의 여러 후보에서 같은 문자열이 나와도 한 번으로 센다.
        distinct = {(r['timestamp_s'], r['text']) for r in rows if r['text']}
        counts = Counter(text for _, text in distinct)
        result.append({'track_id': track, 'first_candidate_s': min(r['timestamp_s'] for r in rows),
            'last_candidate_s': max(r['timestamp_s'] for r in rows),
            'text_frame_counts': dict(counts),
            'repeated_texts': [t for t, n in counts.items() if n >= 3],
            'note': '3개 이상 시점에서 관측. 연속 일치/정답/최종 확정을 뜻하지 않음'})
    return result


def annotate(rgb, buses, predictions, timestamp, fps):
    canvas = rgb.copy()
    scale = min(1., 960 / max(canvas.shape[:2]))
    for bus in buses:
        x1, y1, x2, y2 = bus['bus_box']
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (60,200,255), 3)
        label = f"bus {bus['track_id'] if bus['track_id'] is not None else 'untracked'}"
        cv2.putText(canvas, label, (x1,max(28,y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60,200,255), 2)
        a,b,c,d = bus['search_box']
        cv2.rectangle(canvas, (a,b), (c,d), (255,180,50), 2)
    for p in predictions:
        x1,y1,x2,y2 = p['text_box']
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (100,255,100), 2)
        cv2.putText(canvas, p['text'], (x1,max(25,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100,255,100), 2)
    w, h = [max(2, int(n*scale)//2*2) for n in (canvas.shape[1], canvas.shape[0])]
    canvas = cv2.resize(canvas, (w,h))
    cv2.rectangle(canvas, (0,0), (w,62), (15,15,15), -1)
    cv2.putText(canvas, f't={timestamp:.2f}s | sampled {fps:g}fps | NEW OCR', (8,23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    texts = ' | '.join(f"#{p['track_id']}:{p['text']}" for p in predictions)
    cv2.putText(canvas, (texts or ('NO TEXT CANDIDATE' if buses else 'NO BUS'))[:110], (8,48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,255,100), 1)
    return canvas


def analyze(video, out, detector, recognizer, fps):
    out.mkdir()
    detector.reset()
    frames, predictions = [], []
    reader = sample_frames(video, fps)
    meta = next(reader)
    writer = None
    started = time.perf_counter()
    try:
        for index, (timestamp, rgb) in enumerate(reader):
            detector.sync()
            start = time.perf_counter()
            buses = detector.buses(rgb)
            detector.sync()
            bus_ms = (time.perf_counter()-start)*1000
            current = []
            craft_ms, ocr_ms = 0., 0.
            for bus_index, bus in enumerate(buses):
                begin = time.perf_counter()
                search_box, regions = detector.text_regions(rgb, bus['bus_box'])
                detector.sync()
                craft_ms += (time.perf_counter()-begin)*1000
                bus['search_box'] = search_box
                bus['candidate_count'] = len(regions)
                for candidate, box in enumerate(regions):
                    x1,y1,x2,y2 = box
                    pred = recognizer.predict(Image.fromarray(rgb[y1:y2,x1:x2]))
                    ocr_ms += pred.elapsed_ms
                    row = {'sample_index': index, 'timestamp_s': timestamp,
                        'track_id': bus['track_id'], 'bus_index': bus_index, 'candidate_index': candidate,
                        'bus_box': bus['bus_box'], 'text_box': box, **asdict(pred)}
                    current.append(row)
            detector.sync()
            total_ms = (time.perf_counter()-start)*1000
            frames.append({'sample_index': index, 'timestamp_s': timestamp,
                'status': 'ocr' if current else 'no_text_candidate' if buses else 'no_bus',
                'bus_count': len(buses), 'candidate_count': len(current),
                'bus_ms': bus_ms, 'craft_ms': craft_ms, 'ocr_ms': ocr_ms, 'pipeline_ms': total_ms,
                'buses': buses})
            predictions.extend(current)
            canvas = annotate(rgb, buses, current, timestamp, fps)
            if writer is None:
                writer = video_writer(out/'annotated.mp4', (canvas.shape[1],canvas.shape[0]), fps)
            writer.send(canvas.tobytes())
            if index % 25 == 0:
                print(f'{video.name}: t={timestamp:.1f}s buses={len(buses)} candidates={len(current)}', flush=True)
    finally:
        reader.close()
        if writer is not None:
            writer.close()
    if not frames:
        raise RuntimeError(f'분석 프레임 없음: {video}')
    elapsed = time.perf_counter()-started
    save_csv(out/'frames.csv', frames, list(frames[0]))
    save_csv(out/'predictions.csv', predictions, ['sample_index','timestamp_s','track_id','bus_index',
        'candidate_index','bus_box','text_box','text','token_scores','elapsed_ms'])
    times = [f['pipeline_ms'] for f in frames]
    summary = {'video': str(video.relative_to(ROOT)) if video.is_relative_to(ROOT) else str(video),
        'metadata': meta, 'sample_count': len(frames), 'prediction_count': len(predictions),
        'status_counts': dict(Counter(f['status'] for f in frames)), 'tracks': summarize_tracks(predictions),
        'pipeline_ms': {'p50': float(np.percentile(times,50)), 'p95': float(np.percentile(times,95)), 'max':max(times)},
        'wall_seconds_including_decode_render_encode': elapsed,
        'sample_processing_fps_including_io': len(frames)/elapsed,
        'bus_track_ids': sorted({b['track_id'] for f in frames for b in f['buses'] if b['track_id'] is not None}),
        'evaluation': '개발용 자동 문자 후보. 정답/전면/측면/실제 접근 사건 대응은 별도 검수 필요.',
        'sampling': 'FFmpeg fps filter의 0 기준 시간 격자. 생략한 원본 프레임은 미추론이며 미검출로 집계하지 않음.',
        'latency_scope': 'pipeline_ms는 검출/추적/crop/인식 포함; decode/render/encode/model load 제외. 실시간 전송 지연 아님.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--video', type=Path)
    inputs.add_argument('--videos-csv', type=Path, help='video_path 열이 있는 실행 대상 목록')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--sample-fps', type=float, default=5.)
    parser.add_argument('--top-ratio', type=float, default=.35)
    args = parser.parse_args()
    if not np.isfinite(args.sample_fps) or not 0 < args.sample_fps <= 60:
        parser.error('sample-fps는 0 초과 60 이하여야 합니다.')
    if not np.isfinite(args.top_ratio) or not 0 < args.top_ratio <= 1:
        parser.error('top-ratio는 0 초과 1 이하여야 합니다.')
    if args.output.exists():
        parser.error('기존 output을 덮어쓰지 않습니다.')
    if args.video:
        videos = [args.video.resolve()]
    else:
        with args.videos_csv.open(encoding='utf-8-sig',newline='') as f:
            rows = list(csv.DictReader(f))
        videos = [(ROOT / row['video_path']).resolve() for row in rows]
    if not videos or len(set(videos)) != len(videos) or any(not p.is_file() for p in videos):
        parser.error('영상 목록이 비어 있거나 중복/존재하지 않는 경로가 있습니다.')
    torch.set_num_threads(4)
    detector = BusTextDetector(args.device, args.top_ratio)
    recognizer = PARSeqRecognizer(args.device)
    args.output.mkdir(parents=True)
    settings = {'sample_fps':args.sample_fps,'top_ratio':args.top_ratio,'yolo_imgsz':640,
        'craft_canvas':1280,'craft_magnification':1.5,'craft_thresholds':[.7,.4,.4],
        'craft_commit':CRAFT_COMMIT, 'craft_compatibility':'torchvision VGG weights API patch',
        'python':platform.python_version(),'recognizer':recognizer.metadata(),
        'model_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ROOT/'.cache/yolo11n.pt',ROOT/'.cache/craft_mlt_25k.pth']},
        'videos':[str(v) for v in videos]}
    (args.output/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
    (args.output/'environment.txt').write_text(subprocess.check_output(
        [str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True))
    summaries = []
    for index, video in enumerate(videos,1):
        print(f'[{index}/{len(videos)}] {video.name}',flush=True)
        summaries.append(analyze(video,args.output/f'video_{index:02d}',detector,recognizer,args.sample_fps))
    (args.output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    print(f'완료: {args.output}',flush=True)


if __name__ == '__main__':
    main()
