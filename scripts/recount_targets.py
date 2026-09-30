"""저장된 원시 예측을 사용자 승인 기준(마을버스 지역 접두 생략)으로 재집계."""
import argparse
import csv
import json
from pathlib import Path
from bus_route.target import exact_route


def recount(run):
    with (run/'inputs.csv').open(encoding='utf-8-sig') as f:inputs=list(csv.DictReader(f))
    rows=[]
    for i,item in enumerate(inputs,1):
        with (run/f'video_{i:02d}/predictions.csv').open(encoding='utf-8-sig') as f:preds=list(csv.DictReader(f))
        literal={r['timestamp_s'] for r in preds if r['text']==item['route_hint']}
        accepted={r['timestamp_s'] for r in preds if exact_route(r['text'],item['route_hint'])}
        rows.append({'source_run':run.name,'video_path':item['video_path'],'target_route':item['route_hint'],
                     'literal_samples':len(literal),'accepted_samples':len(accepted),
                     'literal_match':bool(literal),'accepted_match':bool(accepted)})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('기존 출력 덮어쓰기 금지')
    rows=[row for run in args.runs for row in recount(run)];args.output.mkdir(parents=True)
    with (args.output/'recount.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries={run.name:{'videos':sum(r['source_run']==run.name for r in rows),
               'literal_match_videos':sum(r['literal_match'] for r in rows if r['source_run']==run.name),
               'accepted_match_videos':sum(r['accepted_match'] for r in rows if r['source_run']==run.name)} for run in args.runs}
    (args.output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2));print(summaries)


if __name__=='__main__':main()
