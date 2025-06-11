import time 
from collections import deque 
from .logger import EventLogger 

class PSPGazeMetrics:
    # buffers the gaze, detects saccades and jitters on both axes 
    # storing structure - (t_start, t_end, amp, peak_vel, axis)
    
    def __init__(
         self, 
         gaze, 
         history_len: int = 30, 
         vel_thresh: float = 0.5, 
         jitter_thresh: float = 0.05, 
         blink_skip_frames: int = 3, # ignores 3 frames after a blink
         debug: bool = False,
         logger=None, 
         save_on_exit=True
    ):
        self.gaze = gaze
        self.buf = deque(maxlen=history_len)
        self.vel_thresh = vel_thresh
        self.jitter_thresh = jitter_thresh
        self.blink_skip_frames = blink_skip_frames
        self._blink_cooldown = 0
        self.debug = debug
        self.logger = logger or EventLogger()
        self.save_on_exit = save_on_exit
        
        # logging
        self.vert_saccades = []
        self.horiz_saccades = []
        self.jitters = []
        
    def update (self, frame):
        # calls during each video frame and returns a dictionary of current ratios and prev detected events 
        self.gaze.refresh(frame)
        
        # cooldown after blinks 
        if self.gaze.is_blinking():
            self._blink_cooldown = self.blink_skip_frames
            return self._snapshot(None, None)
        
        if self._blink_cooldown > 0:
            self._blink_cooldown -= 1
            return self._snapshot(None, None)
        
        t = time.time()
        h = self.gaze.horizontal_ratio()
        v = self.gaze.vertical_ratio()
        
        if h is None or v is None:
            return self._snapshot(h, v) # ie, missing data
        self.logger.log_frame(t, h, v, self.gaze.is_blinking())
        
        if self.buf:
            self._check_axis(t, "H", h)
            self._check_axis(t, "V", v)
        self.buf.append((t, h, v))
        return self._snapshot(h, v)
        
    # helper functions
    def _snapshot(self, h, v):
        return dict(
            timestamp=time.time(),
            h_ratio=h,
            v_ratio=v,
            last_vert_saccade=self.vert_saccades[-1] if self.vert_saccades else None,
            last_horiz_saccade=self.horiz_saccades[-1] if self.horiz_saccades else None,
            last_jitter=self.jitters[-1] if self.jitters else None,
            blink=self.gaze.is_blinking(),
        )
    
    def _check_axis(self, t, axis, val):
        t0, h0, v0 = self.buf[-1]
        prev = h0 if axis == "H" else v0
        if prev is None:
            return

        dt = t - t0
        if dt <= 0:
            return

        dv = val - prev
        vel = dv / dt
        amp = abs(dv)

        # debug print
        if self.debug:
            print(f"[{axis}] Δ={dv:.3f} dt={dt:.3f} vel={vel:.3f}")

        # classify
        if abs(vel) > self.vel_thresh:
            rec = (t0, t, amp, vel, axis)
            self.logger.log_event(t0, t, amp, vel, axis, "SACCADE")
            if axis == "H":
                self.horiz_saccades.append(rec)
            else:
                self.vert_saccades.append(rec)
            if self.debug:
                print(f"[{axis}] SACCADE amp={amp:.3f} vel={vel:.3f}")
        elif abs(vel) > self.jitter_thresh:
            self.jitters.append((t, vel, axis))
            self.logger.log_event(t0, t, amp, vel, axis, "JITTER")
            if self.debug:
                print(f"[{axis}] JITTER vel={vel:.3f}")