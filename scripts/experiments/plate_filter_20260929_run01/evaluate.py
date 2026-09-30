"""Compare fresh video inference with the fixed single-observation baseline."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np
from bus_route.target import TargetMatcher, exact_route, token_quality

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'output/plate_filter_20260929_run01'
BASE = ROOT/'output/single_20260929_run01'


def read(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(folder):
    frames, predictions = read(folder/'frames.csv'), read(folder/'predictions.csv')
    by = defaultdict(list)
    for frame in frames:
        frame['buses'] = json.loads(frame['buses'])
    for p in predictions:
        p['eligible'] = p['eligible'] == 'True'
        for field in ['token_scores', 'text_box', 'bus_box', 'search_box']:
            p[field] = json.loads(p[field])
        p['context_checks'] = json.loads(p.get('context_checks') or '[]')
        by[p['sample_index']].append(p)
    return frames, predictions, by


def replay(data, target):
    frames, predictions, by = data
    matcher = TargetMatcher(target, single_score=.98)
    for frame in frames:
        for index, bus in enumerate(frame['buses']):
            decision = matcher.update(bus['track_id'], float(frame['timestamp_s']),
                [p for p in by[frame['sample_index']] if int(p['bus_index']) == index],
                bus['bus_box'], complete=bus['decision_complete'])
            if decision['state'] in {'recognized_single', 'matched_candidate'}:
                return {'first_s':float(frame['timestamp_s']), 'track_id':bus['track_id'],
                        'state':decision['state']}
    return None


def stats(values):
    return {'p50':float(np.percentile(values,50)), 'p95':float(np.percentile(values,95)),
            'max':float(max(values))} if values else None


def evaluate(mode):
    source = OUT/mode
    summaries = json.loads((source/'summary.json').read_text())
    assert len(summaries) == 144
    baseline = read(BASE/'comparison.csv')
    reviews = json.loads((BASE/'evidence/visual_review.json').read_text())
    old_flags = json.loads((BASE/'offline/alternative_flags.json').read_text())
    known = {s['target_route'] for s in summaries[:143]}
    rows, flags, diagnostic, preserved, issues = [], [], [], [], []
    newly_filtered_targets = []
    stages = defaultdict(list)
    reasons = Counter()
    total_cases = context_calls = over_age = incomplete = dropped = 0
    for i, (s, old) in enumerate(zip(summaries, baseline), 1):
        folder = f'video_{i:02d}'
        assert s['video'] == old['video_path'] and s['target_route'] == old['target']
        data = load(source/folder)
        recognized = s['first_recognized_s'] is not None
        old_recognized = old[f'{mode}_new'] == 'True'
        row = {'folder':folder, 'video_path':s['video'], 'target':s['target_route'],
               'old_recognized':old_recognized, 'new_recognized':recognized,
               'first_s':s['first_recognized_s'], 'first_repeat_s':s['first_matched_candidate_s'],
               'sample_count':s['sample_count'], 'dropped_samples':s['dropped_samples']}
        rows.append(row)
        if i > 143:
            continue
        dropped += s['dropped_samples']
        for f in data[0]:
            for field in ['pipeline_ms','frame_age_ms','bus_ms','craft_ms','ocr_ms']:
                if f.get(field): stages[field].append(float(f[field]))
            over_age += bool(f.get('frame_age_ms') and float(f['frame_age_ms']) > 300)
            incomplete += sum(not b['decision_complete'] for b in f['buses'])
        for p in data[1]:
            reasons[p['rejection_reason'] or 'eligible'] += 1
            context_calls += len(p['context_checks'])
        for review in [r for r in reviews if r['folder'] == folder]:
            target = review['target']
            raw = [p for p in data[1] if exact_route(p['text'],target)]
            if review['kind'] == 'alternative':
                first = replay(data, target)
                diagnostic.append({'id':review['id'], 'folder':folder, 'target':target,
                    'verdict':review['verdict'], 'raw_ocr_samples':len(raw),
                    'eligible_samples':sum(p['eligible'] for p in raw),
                    'recognized':first is not None, 'first':first,
                    'rejection_reasons':dict(Counter(p['rejection_reason'] or 'eligible' for p in raw))})
            else:
                anchors = [p for p in raw if abs(float(p['timestamp_s'])-review['first_s']) < 1e-6]
                preserved.append({'id':review['id'], 'folder':folder, 'target':target,
                    'anchor_s':review['first_s'], 'anchor_raw':len(anchors),
                    'anchor_eligible':sum(p['eligible'] for p in anchors),
                    'recognized':recognized, 'reasons':[p['rejection_reason'] for p in anchors]})
        # Use the union of old/new OCR strings so filtering does not remove test targets.
        base_ps = read(ROOT/'output/hold_fix_20260929_run01'/folder/'predictions.csv')
        old_eligible = {(p['timestamp_s'],tuple(json.loads(p['text_box'])),p['text'])
                        for p in base_ps if p['eligible']=='True'}
        for p in data[1]:
            if (exact_route(p['text'],s['target_route'])
                    and p['rejection_reason'] in {'likely_registration_plate','embedded_phone_number'}
                    and (p['timestamp_s'],tuple(p['text_box']),p['text']) in old_eligible):
                newly_filtered_targets.append({'folder':folder,'video_path':s['video'],**p})
        alternatives = set(known)
        key = re.sub('^[가-힣]+','',s['target_route'])
        for j,c in enumerate(key):
            if c.isdigit():
                for delta in [-1,1]: alternatives.add(key[:j]+str((int(c)+delta)%10)+key[j+1:])
        for p in [*base_ps, *data[1]]:
            if re.fullmatch(r'(?:M|N)?[0-9]{2,4}[AB]?',p['text']): alternatives.add(p['text'])
        eligible_texts = {p['text'] for p in data[1] if p['eligible'] and token_quality(p['token_scores']) >= .9}
        for target in sorted(alternatives):
            if exact_route(target,s['target_route']): continue
            total_cases += 1
            if not any(exact_route(text,target) for text in eligible_texts): continue
            first = replay(data,target)
            if first:
                old_passed = any(f['folder']==folder and f['target']==target for f in old_flags)
                flags.append({'folder':folder,'video_path':s['video'],'target':target,
                              **first,'baseline_passed':old_passed})
        if old_recognized != recognized:
            issues.append({**row,'kind':'gain' if recognized else 'loss'})
        if i % 20 == 0: print(mode, i, flush=True)
    summary = {'mode':mode,'videos':143,'old_recognized':sum(r['old_recognized'] for r in rows[:143]),
               'new_recognized':sum(r['new_recognized'] for r in rows[:143]),
               'gains':[r for r in issues if r['kind']=='gain'],
               'losses':[r for r in issues if r['kind']=='loss'],
               'negative_control':rows[-1], 'processed_frames':len(stages['pipeline_ms']),
               'dropped_samples':dropped,'frame_age_over_300ms_samples':over_age,
               'incomplete_bus_decisions':incomplete,'timing':{k:stats(v) for k,v in stages.items()},
               'context_ocr_calls':context_calls,'rejection_reasons':dict(reasons),
               'newly_filtered_target_observations':newly_filtered_targets,
               'alternative_cases':total_cases,'alternative_passed':len(flags),
               'new_alternative_flags':[f for f in flags if not f['baseline_passed']],
               'reviewed_diagnostics':diagnostic,'protected_anchors':preserved,
               'slowest_videos':sorted([{'folder':f'video_{i:02d}','video':s['video'],
                    'pipeline_ms':s['pipeline_ms'],'frame_age_ms':s['replay_frame_age_ms'],
                    'dropped_samples':s['dropped_samples']} for i,s in enumerate(summaries[:143],1)],
                    key=lambda s:s['pipeline_ms']['p95'], reverse=True)[:10]}
    write(OUT/f'{mode}_comparison.csv',rows)
    (OUT/f'{mode}_evaluation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/f'{mode}_alternative_flags.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2))
    print(json.dumps({k:summary[k] for k in ['mode','old_recognized','new_recognized','gains','losses','negative_control','alternative_cases','alternative_passed','new_alternative_flags']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['offline','realtime'])
    evaluate(parser.parse_args().mode)
