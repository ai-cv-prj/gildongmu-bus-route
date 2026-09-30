"""회전/HDR 정상화 및 일정 시간 간격의 영상 입출력."""
from pathlib import Path
import subprocess
import imageio_ffmpeg
import numpy as np


def color_filters(video: Path, color: str = 'auto') -> tuple[str, str]:
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', str(video)],
                           capture_output=True, text=True)
    hdr = 'arib-std-b67' in probe.stderr or 'smpte2084' in probe.stderr
    mode = ('hdr' if hdr else 'sdr') if color == 'auto' else color
    filters = []
    if mode == 'hdr':
        filters.append('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,'
                       'tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=full')
    filters.append('format=rgb24')
    return ','.join(filters), mode


def sample_frames(video: Path, fps: float):
    """시각은 FFmpeg fps 필터 출력 기준이며 원본 프레임 PTS와 다를 수 있다."""
    filters, mode = color_filters(video)
    reader = imageio_ffmpeg.read_frames(str(video), output_params=[
        '-vf', f'{filters},fps=fps={fps}:start_time=0', '-threads', '2'])
    try:
        meta = next(reader)
        yield {**meta, 'color_mode': mode, 'sample_fps': fps}
        width, height = meta['size']
        for index, frame in enumerate(reader):
            yield index / fps, np.frombuffer(frame, np.uint8).reshape(height, width, 3).copy()
    finally:
        reader.close()


def video_writer(path: Path, size: tuple[int, int], fps: float):
    writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
        pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=2,
        output_params=['-crf', '23', '-preset', 'fast', '-threads', '2'])
    next(writer)
    return writer


def latest_frames(video: Path, fps: float):
    """원래 재생 시각으로 투입, 최신 1장만 유지. 모델 지연 시 오래된 프레임 폐기."""
    import queue
    import threading
    import time
    reader=sample_frames(video,fps)
    meta=next(reader)
    meta.update(replay_origin=None, last_sample_s=None, replay_mode='latest_one_paced')
    pending=queue.Queue(maxsize=1)
    stop=threading.Event();done=threading.Event();errors=[]
    def produce():
        try:
            origin=time.perf_counter()
            meta['replay_origin']=origin
            for timestamp,rgb in reader:
                if stop.wait(max(0,origin+timestamp-time.perf_counter())):break
                meta['last_sample_s']=timestamp
                try:pending.put_nowait((timestamp,rgb))
                except queue.Full:
                    try:pending.get_nowait()
                    except queue.Empty:pass
                    try:pending.put_nowait((timestamp,rgb))
                    except queue.Full:pass
        except Exception as exc:errors.append(exc)
        finally:
            reader.close();done.set()
    thread=threading.Thread(target=produce,daemon=True)
    try:
        yield meta
        thread.start()
        while not done.is_set() or not pending.empty():
            try:yield pending.get(timeout=.05)
            except queue.Empty:pass
        if errors:raise errors[0]
    finally:
        stop.set()
        if thread.ident is None:
            reader.close()
        else:
            thread.join(timeout=10)
        if thread.is_alive():raise RuntimeError('영상 공급 스레드 종료 실패')
