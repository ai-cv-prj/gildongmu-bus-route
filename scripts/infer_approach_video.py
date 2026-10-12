"""Opt-in native-frame single-bus service experiment (requires stop context)."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from bus_route.approach import ApproachProfile
from bus_route.detector import ROOT
from bus_route.recognizer import PARSeqRecognizer
from bus_route.route_detector import RouteDisplayDetector
from bus_route.service import ApproachPipeline
from bus_route.target import exact_route, token_quality
from bus_route.video import native_frames


def analyze_approach(video, out, detector, recognizer, target, profile=None, realtime=False,
                     temporal=True, frame_source=None, preselection_collection=False,
                     camera_pan_hold=True, realtime_freshness=True):
    out.mkdir(parents=True, exist_ok=False)
    pipeline = ApproachPipeline(detector,recognizer,target,profile,temporal=temporal,
                                preselection_collection=preselection_collection,
                                camera_pan_hold=camera_pan_hold,
                                realtime_freshness=realtime_freshness)
    frames = native_frames(video, max_fps=60) if frame_source is None else frame_source
    times = []
    ages = []
    selected = 0
    ocr = 0
    ocr_before_selection = 0
    ocr_on_selected_bus = 0
    dropped = 0
    first = None
    first_source = None
    decision_states = {}
    started = time.perf_counter()
    origin = None
    with (out/'events.jsonl').open('x') as stream:
        try:
            for frame in frames:
                if origin is None:
                    origin = time.perf_counter()-frame['timestamp_s']
                if realtime:
                    delay = origin+frame['timestamp_s']-time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    if time.perf_counter()-origin-frame['timestamp_s'] > .3:
                        dropped += 1
                        continue
                    # Optional time budget for demand-driven retrospective
                    # detection. The deadline still belongs to this source
                    # frame; old evidence never gains a fresh clock.
                    frame['wall_deadline_monotonic'] = origin+frame['timestamp_s']+.3
                event = pipeline.process(frame)
                age = (time.perf_counter()-origin-frame['timestamp_s'])*1000 if realtime else None
                event['frame_age_ms'] = age
                # Only evidence from the selected bus can trigger a decision.
                # OCR from another bus may be diagnostic but must not make a
                # selected-bus decision appear stale.
                selected_predictions = [p for p in event['predictions']
                                        if p['track_id'] == event['selected_track_id']]
                evidence_age = max((time.perf_counter()-origin-p['timestamp_s']
                                    for p in selected_predictions),default=(age or 0)/1000)*1000 if realtime else None
                event['evidence_age_ms'] = evidence_age
                if realtime and max(age,evidence_age) > 300 and event['decision']:
                    event['decision'] = {'state':'hold','reason':'stale_result','support_samples':0,'conflicts':[]}
                    # Late OCR must not support a future result.
                    pipeline.matcher.tracks.clear()
                if event['decision'] and event['decision']['state'] in {'recognized_single','matched_candidate'} and first is None:
                    first = event['timestamp_s']
                    supports = [p['timestamp_s'] for p in selected_predictions
                                if p['eligible'] and exact_route(p['text'], target)
                                and token_quality(p.get('token_scores', [])) >= .9]
                    first_source = max(supports, default=None)
                selected += event['selected_track_id'] is not None
                ocr += len(event['predictions'])
                if event['selected_track_id'] is None:
                    ocr_before_selection += len(event['predictions'])
                ocr_on_selected_bus += len(selected_predictions)
                if event['decision']:
                    state = event['decision']['state']
                    decision_states[state] = decision_states.get(state, 0) + 1
                times.append(event['pipeline_ms'])
                if age is not None:
                    ages.append(age)
                stream.write(json.dumps(event,ensure_ascii=False)+'\n')
        finally:
            close = getattr(frames,'close',None)
            if close:
                close()
    summary = {'video':str(video),'target_route':target,'mode':'realtime' if realtime else 'offline',
               'temporal':temporal,'preselection_collection':preselection_collection,
               'camera_pan_hold':camera_pan_hold,'realtime_freshness':realtime_freshness,
               'frames':len(times),'selected_frames':selected,'ocr_predictions':ocr,
               'ocr_before_selection':ocr_before_selection,
               'ocr_on_selected_bus':ocr_on_selected_bus,'decision_states':decision_states,
               'dropped_frames':dropped,'first_recognized_s':first,
               'first_candidate_source_s':first_source,
               'pipeline_p95_ms':float(np.percentile(times,95)) if times else None,
               'frame_age_p95_ms':float(np.percentile(ages,95)) if ages else None,
               'wall_s':time.perf_counter()-started,
               'profile':asdict(profile) if profile else None,
               'limitations':['Configured context; not an automatic lane/door classifier.',
                              'Relative stroke coverage is not certified full-number readability.',
                              'Route model and OCR retain known fleet-number/ad/Hangul limitations.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-route',required=True)
    parser.add_argument('--profile',type=Path,help='Normalized lane/approach context JSON; absent -> no OCR')
    parser.add_argument('--route-weights',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--realtime',action='store_true')
    parser.add_argument('--preselection-collection',action='store_true',
                        help='Enable experimental selected-bus retrospective evidence collection')
    parser.add_argument('--no-camera-pan-hold',action='store_true',
                        help='Baseline: reset selected bus on all measured camera motion')
    parser.add_argument('--no-realtime-freshness',action='store_true',
                        help='Baseline: rank source crops without the live 300 ms budget')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing output must not be overwritten')
    profile = ApproachProfile(**json.loads(args.profile.read_text())) if args.profile else None
    torch.set_num_threads(4)
    detector = RouteDisplayDetector(args.route_weights,args.device)
    recognizer = PARSeqRecognizer(args.device)
    analyze_approach(args.video,args.output,detector,recognizer,args.target_route,profile,args.realtime,
                     preselection_collection=args.preselection_collection,
                     camera_pan_hold=not args.no_camera_pan_hold,
                     realtime_freshness=not args.no_realtime_freshness)
    (args.output/'provenance.json').write_text(json.dumps({'route_weights_sha256':hashlib.sha256(args.route_weights.read_bytes()).hexdigest(),
        'recognizer':recognizer.metadata(),'profile_sha256':hashlib.sha256(args.profile.read_bytes()).hexdigest() if args.profile else None},indent=2))


if __name__ == '__main__':
    main()
