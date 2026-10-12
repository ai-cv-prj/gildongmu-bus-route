"""Camera-configured approach selection for a UI replay, not a road/lane classifier."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Profile:
    entry_right: float = .30
    entry_bottom: float = .55
    approach_seconds: float = .5
    approach_growth: float = 1.20
    max_gap: float = .6
    exit_right: float = .98
    exit_center: float = .50
    exit_area: float = .22
    read_timeout: float = 1.2

class Selector:
    def __init__(self, width, height, profile=None):
        self.w=width; self.h=height; self.profile=profile or Profile()
        self.history={}; self.retired=set(); self.focus=None; self.started=None
        self.number=None; self.box=None; self.last_time=None; self.events=[]
    def _geometry(self, box):
        x,y,r,b=box
        return ((x+r)/2/self.w,b/self.h,(r-x)*(b-y)/(self.w*self.h),r/self.w)
    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]); y=max(a[1],b[1]); r=min(a[2],b[2]); d=min(a[3],b[3])
        inter=max(0,r-x)*max(0,d-y)
        return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    def _retire(self,t,reason):
        if self.focus is not None:
            self.events.append(dict(time=t,track_id=self.focus,number=self.number,reason=reason));self.retired.add(self.focus)
        self.focus=None;self.started=None;self.number=None;self.box=None;self.last_time=None
    def update(self,t,buses):
        p=self.profile
        current={b['track_id']:b for b in buses}
        for key,b in current.items():
            hist=self.history.setdefault(key,[])
            if hist and t-hist[-1][0]>p.max_gap:hist.clear()
            hist.append((t,b['bus_box']));self.history[key]=[(ts,box) for ts,box in hist if t-ts<=1.4]
        if self.focus is not None:
            bus=current.get(self.focus)
            if bus is None:
                self._retire(t,'track_disappeared')
            elif t-self.last_time>p.max_gap or self._iou(self.box,bus['bus_box'])<.05:
                self._retire(t,'identity_continuity_lost')
            else:
                cx,bottom,area,right=self._geometry(bus['bus_box'])
                if right>=p.exit_right and cx>=p.exit_center and area>=p.exit_area:
                    self._retire(t,'left_to_right_exit_zone')
        if self.focus is None:
            candidates=[]
            for key,b in current.items():
                if key in self.retired:continue
                cx,bottom,area,right=self._geometry(b['bus_box']);hist=self.history[key]
                if not (0<=cx<=p.entry_right and 0<=bottom<=p.entry_bottom):continue
                if len(hist)<3 or t-hist[0][0]<p.approach_seconds:continue
                oldcx,_,oldarea,_=self._geometry(hist[0][1])
                if area<oldarea*p.approach_growth or cx<oldcx:continue
                candidates.append((area,key))
            if candidates:
                _,self.focus=max(candidates);self.started=t
                self.events.append(dict(time=t,track_id=self.focus,reason='approaching_in_configured_entry'))
        if self.focus is not None:
            self.box=current[self.focus]['bus_box'];self.last_time=t
        return current.get(self.focus)
    def observe(self,t,numbers):
        if self.focus is None:return
        if self.number is not None and any(n!=self.number for n in numbers):
            self._retire(t,'conflicting_full_number');return
        if len(numbers)==1 and self.number is None:
            self.number=numbers[0]
            self.events.append(dict(time=t,track_id=self.focus,number=self.number,reason='number_read'))
    def state(self,t):
        if self.focus is None:return 'search'
        if self.number is not None:return 'recognized'
        return 'failed' if t-self.started>=self.profile.read_timeout else 'checking'
