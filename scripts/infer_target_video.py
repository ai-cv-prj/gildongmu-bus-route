"""상단+차체 탐색과 보수적인 목표 노선 판단. 원시 OCR에 목표 번호를 주입하지 않는다."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch
from bus_route.detector import ROOT,BusTextDetector,CRAFT_COMMIT
from bus_route.recognizer import PARSeqRecognizer
from bus_route.target import TargetMatcher,exact_route,route_aliases
from bus_route.route_evidence import (text_rejection, context_boxes, context_rejection,
                                      EVIDENCE_FILTER)
from bus_route.video import sample_frames,latest_frames,video_writer
# 프레임 단위 근거 규칙은 통합앱 런타임과 같은 bus_route.evidence를 공유한다.
from bus_route.evidence import (DUPLICATE_BUS_POLICY, FALLBACK_EVIDENCE_POLICY,
                                gate_fallback_evidence, recover_duplicate_bus_candidate,
                                eligible_box, select_candidates, recover_regions)
from bus_route.region_recovery import RECOVERY_POLICY
from infer_video import annotate,save_csv,summarize_tracks


def analyze(video,out,detector,recognizer,fps,target,alternatives=(),realtime=False,body_interval=.2,max_candidates=24,single_score=.98,craft_fallback_on_empty=False):
    if craft_fallback_on_empty and not hasattr(detector,'craft_fallback_on_empty'):
        raise ValueError('CRAFT fallback requires a route-display detector')
    if hasattr(detector,'craft_fallback_on_empty'):
        detector.craft_fallback_on_empty=craft_fallback_on_empty
    out.mkdir();detector.reset();matcher=TargetMatcher(target,alternatives,single_score=single_score)
    frames=[];predictions=[];decisions=[];last_body={};writer=None
    reader=(latest_frames if realtime else sample_frames)(video,fps);meta=next(reader)
    started=time.perf_counter();render_index=0
    try:
        for timestamp,rgb in reader:
            index=round(timestamp*fps)
            detector.sync();begin=time.perf_counter();buses=detector.buses(rgb);detector.sync()
            bus_ms=(time.perf_counter()-begin)*1000;craft_ms=0.;pending=[]
            for j,bus in enumerate(buses):
                t=time.perf_counter();upper,boxes=detector.text_regions(rgb,bus['bus_box'])
                region_source=getattr(detector,'last_region_source','craft')
                bus['search_box']=upper;bus['body_search_box']=None
                bus['region_source']=region_source
                items=[{'text_box':box,'search_zone':'upper','search_box':upper,'region_source':region_source} for box in boxes]
                identity=bus['track_id']
                if identity is None or timestamp-last_body.get(identity,-1e9)>=body_interval-1e-6:
                    x,y,z,w=bus['bus_box'];body=(x,y+int((w-y)*.25),z,w)
                    bodyboxes=detector.regions_in_box(rgb,body,canvas_size=960)
                    bus['body_search_box']=body
                    if identity is not None:last_body[identity]=timestamp
                    for box in bodyboxes:
                        items.append({'text_box':box,'search_zone':'body','search_box':body,'region_source':region_source})
                detector.sync();craft_ms+=(time.perf_counter()-t)*1000
                selected,count,valid_count,truncated=select_candidates(rgb,items,j,buses,max_candidates)
                bus['candidate_count']=count;bus['eligible_candidate_count']=valid_count
                bus['candidate_truncated']=truncated
                for k,item in enumerate(selected):
                    pending.append({'sample_index':index,'timestamp_s':timestamp,'track_id':identity,
                                    'bus_index':j,'candidate_index':k,'bus_box':bus['bus_box'],**item})
            t=time.perf_counter();current=[]
            # 후보를 소규모 배치로 처리해 GPU 메모리 사용을 제한한다.
            for start in range(0,len(pending),16):
                chunk=pending[start:start+16]
                crops=[Image.fromarray(rgb[p['text_box'][1]:p['text_box'][3],p['text_box'][0]:p['text_box'][2]]) for p in chunk]
                for item,pred in zip(chunk,recognizer.predict_batch(crops)):
                    rejection=item['rejection_reason'] or text_rejection(pred.text,item['text_box'])
                    item={**item,'eligible':not rejection,'rejection_reason':rejection}
                    current.append({**item,**asdict(pred),'duplicate_bus_recovered':False})
            if getattr(detector, 'route_display_only', False):
                recover_duplicate_bus_candidate(rgb,current,buses)
            # 작은 4자리 차체 숫자는 좌우 문맥을 거절 전용으로 확인한다.
            # 원시 OCR과 문맥 OCR을 분리하여 부분 숫자를 목표로 채우지 않는다.
            contexts=[]
            for item in current:
                item['context_checks']=[]
                for box in context_boxes(item,rgb.shape):
                    contexts.append((item,box))
            for start in range(0,len(contexts),16):
                chunk=contexts[start:start+16]
                crops=[Image.fromarray(rgb[b:d,a:c]) for _,(a,b,c,d) in chunk]
                for (item,box),pred in zip(chunk,recognizer.predict_batch(crops)):
                    rejection=context_rejection(item['text'],pred.text,pred.token_scores)
                    item['context_checks'].append({'box':box,**asdict(pred),'rejection_reason':rejection})
                    if rejection:
                        item['eligible']=False
                        item['rejection_reason']=rejection
            recover_regions(rgb,current,buses,recognizer)
            if craft_fallback_on_empty:
                gate_fallback_evidence(current)
            detector.sync();ocr_ms=(time.perf_counter()-t)*1000
            for j,bus in enumerate(buses):
                complete=not bus['candidate_truncated'] and (not realtime or time.perf_counter()-meta['replay_origin']-timestamp<=.3)
                bus['decision_complete']=complete
                decision=matcher.update(bus['track_id'],timestamp,[p for p in current if p['bus_index']==j],
                                        bus['bus_box'],complete=complete)
                bus['target_decision']=decision
                decisions.append({'sample_index':index,'timestamp_s':timestamp,'track_id':bus['track_id'],
                                  'target_route':target,**decision})
            detector.sync();pipeline_ms=(time.perf_counter()-begin)*1000
            age_ms=(time.perf_counter()-meta['replay_origin']-timestamp)*1000 if realtime else None
            frames.append({'sample_index':index,'timestamp_s':timestamp,'status':'ocr' if current else 'no_text_candidate' if buses else 'no_bus',
                           'bus_count':len(buses),'candidate_count':len(current),'bus_ms':bus_ms,'craft_ms':craft_ms,
                           'ocr_ms':ocr_ms,'pipeline_ms':pipeline_ms,'frame_age_ms':age_ms,'buses':buses})
            predictions.extend(current)
            marked=rgb.copy()
            for bus in buses:
                if bus['body_search_box']:
                    a,b,c,d=bus['body_search_box'];cv2.rectangle(marked,(a,b),(c,d),(230,80,240),2)
            canvas=annotate(marked,buses,current,timestamp,fps)
            text=' | '.join(f"#{b['track_id']} {b['target_decision']['state']}" for b in buses) or 'NO BUS'
            cv2.rectangle(canvas,(0,62),(canvas.shape[1],100),(15,15,15),-1)
            cv2.putText(canvas,f'TARGET {target.encode("ascii","ignore").decode()} | '+text[:90],(8,84),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            if writer is None:writer=video_writer(out/'annotated.mp4',(canvas.shape[1],canvas.shape[0]),fps)
            # 폐기한 시점에 과거 예측을 새 예측처럼 표시하지 않는다.
            while render_index<index:
                blank=np.zeros_like(canvas);cv2.putText(blank,f't={render_index/fps:.2f}s NOT INFERRED (dropped)',(8,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                writer.send(blank.tobytes());render_index+=1
            writer.send(canvas.tobytes());render_index+=1
            if index%25==0:print(f'{video.name} t={timestamp:.1f}s candidates={len(current)} {text}',flush=True)
    finally:
        reader.close()
        if writer is not None:writer.close()
    if not frames:raise RuntimeError('분석 프레임 없음')
    wall=time.perf_counter()-started
    save_csv(out/'frames.csv',frames,list(frames[0]))
    save_csv(out/'predictions.csv',predictions,['sample_index','timestamp_s','track_id','bus_index','candidate_index','bus_box','text_box','search_zone','search_box','region_source','eligible','rejection_reason','text','token_scores','elapsed_ms','duplicate_bus_recovered','context_checks','recovery_checks','recovered_from'])
    save_csv(out/'decisions.csv',decisions,['sample_index','timestamp_s','track_id','target_route','state','reason','support_samples','conflicts'])
    times=[r['pipeline_ms'] for r in frames]
    matches=[p for p in predictions if exact_route(p['text'],target)]
    passed=[d for d in decisions if d['state']=='matched_candidate']
    recognized=[d for d in decisions if d['state'] in {'matched_candidate','recognized_single'}]
    age=[r['frame_age_ms'] for r in frames if r['frame_age_ms'] is not None]
    summary={'video':str(video.relative_to(ROOT)),'metadata':meta,'sample_count':len(frames),
             'prediction_count':len(predictions),'status_counts':dict(Counter(f['status'] for f in frames)),
             'tracks':summarize_tracks(predictions),'pipeline_ms':{'p50':float(np.percentile(times,50)),'p95':float(np.percentile(times,95)),'max':max(times)},
             'stage_p95_ms':{key:float(np.percentile([r[key] for r in frames],95)) for key in ['bus_ms','craft_ms','ocr_ms']},
             'wall_seconds_including_decode_render_encode':wall,'sample_processing_fps_including_io':len(frames)/wall,
             'rendered_frame_count':render_index,'dropped_samples':render_index-len(frames),
             'replay_frame_age_ms':{'p50':float(np.percentile(age,50)),'p95':float(np.percentile(age,95)),'max':max(age)} if age else None,
             'target_route':target,'target_match_samples':len({p['timestamp_s'] for p in matches}),
             'first_exact_s':min((p['timestamp_s'] for p in matches),default=None),
             'first_matched_candidate_s':min((p['timestamp_s'] for p in passed),default=None),
             'first_recognized_s':min((p['timestamp_s'] for p in recognized),default=None),
             'first_single_s':min((p['timestamp_s'] for p in decisions if p['state']=='recognized_single'),default=None),
             'body_match_samples':len({p['timestamp_s'] for p in matches if p['search_zone']=='body'}),
             'craft_fallback_on_empty':craft_fallback_on_empty,
             'fallback_evidence_policy':FALLBACK_EVIDENCE_POLICY if craft_fallback_on_empty else None,
             'craft_fallback_bus_frames':sum(b['region_source']=='craft' for f in frames for b in f['buses']) if getattr(detector,'route_display_only',False) else 0,
             'decision_counts':dict(Counter(d['state'] for d in decisions)),
             'evaluation':'정답 미검수 실험용 후보; 부분 숫자 보정 없음; LED/PRINT 또는 전면/측면 자동 분류 없음'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inp=p.add_mutually_exclusive_group(required=True);inp.add_argument('--video',type=Path);inp.add_argument('--videos-csv',type=Path)
    targets=p.add_mutually_exclusive_group(required=True);targets.add_argument('--target-route');targets.add_argument('--target-from-manifest',action='store_true',help='CSV target_route 열: 실험용 사용자 목표 시뮬레이션')
    p.add_argument('--competitors',nargs='*',default=[]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-fps',type=float,default=5.);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--realtime',action='store_true')
    p.add_argument('--single-score',type=float,default=.98,help='전체 문자 및 EOS의 단일 관측 최소 점수; 보정된 확률 아님')
    p.add_argument('--repeat-only',action='store_true',help='기존 3회 반복 판정만 사용')
    args=p.parse_args()
    if args.output.exists():p.error('기존 출력 덮어쓰기 금지')
    if not np.isfinite(args.sample_fps) or not 0<args.sample_fps<=30:p.error('fps 범위: 0 초과 30 이하')
    if args.video:
        if not args.target_route:p.error('단일 영상은 --target-route 필수')
        rows=[{'video_path':str(args.video),'target_route':args.target_route}]
    else:
        with args.videos_csv.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if not rows:p.error('빈 목록')
    for row in rows:
        row['path']=(ROOT/row['video_path']).resolve()
        row['target']=args.target_route or row.get('target_route','')
        TargetMatcher(row['target'],args.competitors,single_score=None if args.repeat_only else args.single_score)
        if not row['path'].is_file():p.error('영상 없음: '+str(row['path']))
    torch.set_num_threads(4);detector=BusTextDetector(args.device);recognizer=PARSeqRecognizer(args.device)
    args.output.mkdir(parents=True)
    settings={'sample_fps':args.sample_fps,'top_ratio':.35,'body_y_start':.25,'body_interval_s':.2,'body_canvas':960,
              'candidate_selection':'pre-OCR eligible first; intact cross-zone crop replaces rejected duplicate; incomplete only for omitted eligible candidates',
              'max_candidates_per_bus':24,'ocr_batch':16,'realtime':args.realtime,'competitors':args.competitors,
              'recognizer':recognizer.metadata(),'craft_commit':CRAFT_COMMIT,'target_mode':'manifest_simulation' if args.target_from_manifest else 'user_target',
              'decision_policy':{'full_match_samples':3,'window_s':2.0,'min_span_s':.4,'min_token_score_including_eos':.9,'conflict_min_score':.75,
                                 'single_min_token_score':None if args.repeat_only else args.single_score,
                                 'recognition_states':['matched_candidate'] if args.repeat_only else ['matched_candidate','recognized_single'],
                                 'partial_completion':False,'max_replay_age_ms':300,'score_note':'검증된 정답 확률 아님',
                                 'same_track_conflict':'recent 2.0s; strong >=.9 or repeated >=.75 veto; single weak requires 3 full matches with distinguishing token >=.97',
                                 'implicit_competitors':'same-length one-character difference or longer prefix/suffix route; implicit longer requires score >=.9, explicit alternatives take precedence'},
              'evidence_filter':EVIDENCE_FILTER,
              'region_recovery':RECOVERY_POLICY,
              'model_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [ROOT/'.cache/yolo11n.pt',ROOT/'.cache/craft_mlt_25k.pth']}}
    (args.output/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
    (args.output/'environment.txt').write_text(subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True))
    summaries=[]
    for i,row in enumerate(rows,1):
        print(f'[{i}/{len(rows)}] target={row["target"]} {row["path"].name}',flush=True)
        summaries.append(analyze(row['path'],args.output/f'video_{i:02d}',detector,recognizer,args.sample_fps,row['target'],args.competitors,args.realtime,single_score=None if args.repeat_only else args.single_score))
    (args.output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
