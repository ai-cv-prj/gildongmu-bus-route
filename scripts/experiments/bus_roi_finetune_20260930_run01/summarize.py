"""Final grouped video and error summaries; does not edit source evaluations."""
import json
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'output/bus_roi_finetune_20260930_run01'
for mode in ['offline','realtime']:
 e=json.loads((O/f'{mode}_comparison.json').read_text());groups=defaultdict(list)
 for r in e['rows'][:143]:
  day=Path(r['video_path']).name[:8];group='test_dates' if day in ['20260918','20260921'] else 'validation_date' if day=='20260920' else 'training_dates';groups[group].append(r)
 out={k:dict(n=len(rs),old=sum(r['old_recognized'] for r in rs),new=sum(r['new_recognized'] for r in rs),gain=[r['folder'] for r in rs if not r['old_recognized'] and r['new_recognized']],loss=[r['folder'] for r in rs if r['old_recognized'] and not r['new_recognized']]) for k,rs in groups.items()}
 (O/f'{mode}_grouped_final.json').write_text(json.dumps(out,indent=2))
 print(mode,json.dumps(out))
if (O/'video_evidence/review.json').exists():
 rows=json.loads((O/'video_evidence/review.json').read_text());assert all(r['verdict']!='needs_review' for r in rows)
 result={mode:dict(new_flags=Counter(r['verdict'] for r in rows if r['mode']==mode and r['kind']=='new_alternative'),physical_new_error_cases=len({(r['folder'],r['time'],tuple(r['box'])) for r in rows if r['mode']==mode and r['kind']=='new_alternative' and r['verdict']=='false_recognition'})) for mode in ['offline','realtime']}
 (O/'review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result)
