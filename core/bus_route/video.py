"""회전/HDR 정상화 및 일정 시간 간격의 영상 입출력."""
from pathlib import Path
import subprocess
import imageio_ffmpeg
import numpy as np


def native_frames(video: Path, max_fps=None):
    """Yield actual decoded frames with source PTS; no fps filter or interpolation.

    FFmpeg applies the same autorotation and HDR mapping as sample_frames. PPM
    and showinfo are paired one-to-one, avoiding container nominal-fps guesses.
    timestamp_s is relative to the first decoded PTS; source_pts_s is preserved.
    """
    import hashlib
    import math
    import queue
    import re
    import threading
    filters, mode = color_filters(video)
    if max_fps is not None and (not math.isfinite(max_fps) or max_fps <= 0):
        raise ValueError('max_fps must be positive')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'info',
               '-copyts', '-i', str(video), '-map', '0:v:0', '-an', '-sn',
               '-vf', filters+',showinfo', '-vsync', '0', '-threads', '2',
               '-c:v', 'ppm', '-f', 'image2pipe', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    infos = queue.Queue()
    errors = []
    pattern = re.compile(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+\-]+)')
    def drain():
        try:
            for line in iter(process.stderr.readline, b''):
                line = line.decode('utf-8', errors='replace')
                match = pattern.search(line)
                if match:
                    infos.put((int(match[1]), int(match[2]), float(match[3])))
                errors.append(line)
                del errors[:-30]
        finally:
            infos.put(None)
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    origin = None
    previous = None
    emitted_time = None
    emitted_digest = None
    try:
        while True:
            magic = process.stdout.readline()
            if not magic:
                break
            if magic != b'P6\n':
                raise RuntimeError('Invalid native RGB frame header')
            w, h = map(int, process.stdout.readline().split())
            if process.stdout.readline().strip() != b'255':
                raise RuntimeError('Invalid native frame depth')
            data = bytearray()
            while len(data) < w*h*3:
                chunk = process.stdout.read(w*h*3-len(data))
                if not chunk:
                    raise RuntimeError('Truncated native frame')
                data.extend(chunk)
            info = infos.get(timeout=30)
            if info is None:
                raise RuntimeError('Native PTS missing: '+''.join(errors))
            index, pts, seconds = info
            if previous is not None and seconds <= previous:
                # Do not fabricate independent observations for duplicate PTS.
                continue
            if origin is None:
                origin = seconds
            previous = seconds
            if max_fps is not None and emitted_time is not None and seconds-emitted_time < 1/max_fps-1e-6:
                continue
            digest = hashlib.blake2b(data, digest_size=16).digest()
            if digest == emitted_digest:
                continue
            emitted_time, emitted_digest = seconds, digest
            yield {'source_frame_index': index, 'source_pts': pts, 'source_pts_s': seconds,
                   'timestamp_s': seconds-origin, 'color_mode': mode,
                   'rgb': np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()}
        if process.wait(timeout=30):
            raise RuntimeError('Native decoding failed: '+''.join(errors))
    finally:
        if process.poll() is None:
            process.terminate()
        process.stdout.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        thread.join(timeout=5)
        process.stderr.close()


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
    # Select the same samples before costly per-frame HDR/color conversion.
    # Keep the fps grid and replay clock unchanged; do not hide decode latency.
    reader = imageio_ffmpeg.read_frames(str(video), output_params=[
        '-vf', f'fps=fps={fps}:start_time=0,{filters}', '-threads', '2'])
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
