"""YOLO11n/ByteTrack 및 공식 CRAFT General 추론. 정답은 입력받지 않는다."""
from pathlib import Path
import os
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CRAFT_COMMIT = 'e332dd8b718e291f51b66ff8f9ef2c98ee4474c8'


def clip_box(box, width, height):
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    return tuple(int(v) for v in (max(0, min(width, x1)), max(0, min(height, y1)),
            max(0, min(width, x2)), max(0, min(height, y2))))


class BusTextDetector:
    def __init__(self, device='cuda', top_ratio=0.35, canvas_size=1280):
        (ROOT / '.cache/ultralytics').mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics'))
        from ultralytics import YOLO
        repo = ROOT / '.cache/CRAFT-pytorch'
        if not (repo / 'craft.py').is_file():
            raise FileNotFoundError('먼저 scripts/setup_video_models.py를 실행하세요.')
        sys.path.insert(0, str(repo))
        from craft import CRAFT
        import craft_utils
        import imgproc
        self.utils, self.imgproc = craft_utils, imgproc
        self.device, self.top_ratio, self.canvas_size = device, top_ratio, canvas_size
        self.yolo = YOLO(str(ROOT / '.cache/yolo11n.pt'))
        self.bus_ids = [i for i, name in self.yolo.names.items() if name == 'bus']
        self.craft = CRAFT(pretrained=False)
        state = torch.load(ROOT / '.cache/craft_mlt_25k.pth', map_location='cpu', weights_only=True)
        self.craft.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
        self.craft.eval().to(device)
        self.yolo.predict(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)
        with torch.inference_mode():
            self.craft(torch.zeros(1, 3, 64, 64, device=device))
        self.sync()

    def sync(self):
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()

    def reset(self):
        # 영상 간 ID/추적 상태를 연결하지 않는다.
        if self.yolo.predictor is not None and hasattr(self.yolo.predictor, 'trackers'):
            for tracker in self.yolo.predictor.trackers:
                tracker.reset()

    def buses(self, rgb):
        result = self.yolo.track(rgb[:, :, ::-1].copy(), persist=True, tracker='bytetrack.yaml',
            classes=self.bus_ids, conf=0.25, imgsz=640, device=self.device, verbose=False)[0]
        h, w = rgb.shape[:2]
        buses = []
        for box in result.boxes:
            bounds = clip_box(box.xyxy[0].cpu().numpy(), w, h)
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            buses.append({'track_id': int(box.id.item()) if box.id is not None else None,
                          'bus_box': bounds, 'bus_score': float(box.conf.item())})
        return buses

    def text_regions(self, rgb, bus_box):
        x1, y1, x2, bottom = bus_box
        y2 = min(bottom, y1 + max(1, int((bottom-y1) * self.top_ratio)))
        return (x1,y1,x2,y2), self.regions_in_box(rgb, (x1,y1,x2,y2))

    def regions_in_box(self, rgb, search_box, canvas_size=None):
        x1,y1,x2,y2 = search_box
        roi = rgb[y1:y2, x1:x2]
        resized, ratio, _ = self.imgproc.resize_aspect_ratio(
            roi, canvas_size or self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        tensor = torch.from_numpy(self.imgproc.normalizeMeanVariance(resized)).permute(2, 0, 1)
        with torch.inference_mode():
            scores, _ = self.craft(tensor.unsqueeze(0).to(self.device))
        score = scores[0].cpu().numpy()
        boxes, _ = self.utils.getDetBoxes(score[:, :, 0], score[:, :, 1], 0.7, 0.4, 0.4, False)
        boxes = self.utils.adjustResultCoordinates(boxes, 1/ratio, 1/ratio)
        regions = []
        for polygon in boxes:
            # 초기 기준: 축 정렬 crop. 후보 합치기·문자 보정은 하지 않는다.
            bounds = clip_box([*polygon.min(axis=0), *polygon.max(axis=0)], x2-x1, y2-y1)
            a, b, c, d = bounds
            if c-a >= 2 and d-b >= 2:
                regions.append((a+x1, b+y1, c+x1, d+y1))
        return regions
