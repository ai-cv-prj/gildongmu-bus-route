"""Fixed route-display model on the selected bus; no full-image coordinate mask."""
import numpy as np
from .detector import BusTextDetector, clip_box


class RouteDisplayDetector(BusTextDetector):
    def __init__(self, weights, device='cuda'):
        super().__init__(device)
        from ultralytics import YOLO
        self.route = YOLO(str(weights))
        self.route.predict(np.zeros((960,960,3),np.uint8),device=device,imgsz=960,
                           rect=False,conf=.25,iou=.5,verbose=False)

    def route_regions(self, rgb, bus_box):
        x,y,z,b = bus_box
        result = self.route.predict(rgb[y:b,x:z,::-1].copy(),device=self.device,
                                    imgsz=960,rect=False,conf=.25,iou=.5,verbose=False)[0]
        boxes = []
        for bounds in result.boxes.xyxy.cpu().tolist():
            a,c,d,e = clip_box(bounds,z-x,b-y)
            if d > a and e > c:
                boxes.append((a+x,c+y,d+x,e+y))
        return boxes
