"""파일명 노선·조건을 검증해 영상 분석 입력 목록을 만든다. 정답은 추론 입력과 분리."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from bus_route.detector import ROOT


def describe_video(path):
    parts = path.stem.split('_')
    if len(parts) < 4:
        raise ValueError(f'파일명 형식 확인 필요: {path.name}')
    index = 3 if parts[2] in {'세로', '가로'} else 2
    route = parts[index]
    if not re.fullmatch(r'(?:[가-힣]+|[A-Za-z]+)?[0-9]+[A-Za-z]?(?:-[0-9]+)?', route):
        raise ValueError(f'노선 해석 불가: {path.name} → {route}')
    return {'video_path':str(path.resolve().relative_to(ROOT)), 'route_hint':route,
            'display_tag':'LED+PRINT' if 'LED' in parts[-1] and 'PRINT' in parts[-1] else
                'LED+PRINT' if any('LED' in p for p in parts) and any('PRINT' in p for p in parts) else
                'LED' if 'LED' in parts else 'PRINT' if 'PRINT' in parts else 'unspecified',
            'view_tag':'side' if '측면' in parts else 'unspecified',
            'condition_tag':'|'.join(t for t in ['빛번짐','빛반사','불량'] if t in parts) or 'unspecified',
            'route_source':f'filename token {index+1}', 'capture_date':parts[0], 'device_tag':parts[1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('기존 목록을 덮어쓰지 않습니다.')
    paths=sorted(p for p in args.videos.rglob('*') if p.suffix.lower() in {'.mp4','.mov','.m4v','.avi','.mkv'})
    if not paths: parser.error('영상이 없습니다.')
    rows=[];excluded=[]
    for path in paths:
        row=describe_video(path)
        if '공항' in path.stem.split('_'):
            excluded.append({**row,'reason':'filename explicitly marks airport bus'})
        else: rows.append(row)
    if not rows: parser.error('분석 대상이 없습니다.')
    args.output.mkdir(parents=True)
    with (args.output/'inputs.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    inventory={'dataset':args.videos.name,'total':len(paths),'included':len(rows),'excluded':excluded,
               'display_counts':dict(Counter(r['display_tag'] for r in rows)),
               'route_counts':dict(Counter(r['route_hint'] for r in rows)),
               'device_counts':dict(Counter(r['device_tag'] for r in rows)),
               'capture_date_counts':dict(Counter(r['capture_date'] for r in rows)),
               'label_note':'파일명 기준; 세로/가로 토큰 뒤의 노선을 지원. 사건 독립성/차종/LED 상태 미검수.'}
    (args.output/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2))
    print(json.dumps(inventory,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
