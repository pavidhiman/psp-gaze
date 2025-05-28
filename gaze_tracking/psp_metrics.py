import time 
from collections import deque 

class PSPGazeMetrics:
    #raw gaze ratios and timesteps, detects vertical saccades and fixation jitters
    
    def __init__(self, gaze, calib, history_len=30, vel_thresh=0.5, jitter_thresh=0.05, debug=False):
         self.gaze = gaze
         self.calib = calib
         self.buf = deque(maxlen=history_len)
         self.vel_thresh = vel_thresh
         self.jitter_thresh = jitter_thresh
         self.debug = debug

         #store events
         self.saccades = []
         self.jitters = []

    
    def update (self, frame):
        #returns dict of current raw ratios and new PSP events by calling each video frame
        self.gaze.refresh(frame)
        t = time.time()
        h = self.gaze.horizontal_ratio()
        v = self.gaze.vertical_ratio()
        
        if self.buf:
            self._check_events(t, v)
        self.buf.append((t, h, v))
        
        return {
            'timestamp': t,
            'h_ratio':    h,
            'v_ratio':    v,
            'saccade':    self.saccades[-1] if self.saccades else None,
            'jitter':     self.jitters[-1] if self.jitters else None,
            'blink':      self.gaze.is_blinking()
        }
        
    def _check_events(self, t, v):
        t0, _, v0 = self.buf[-1]
        
        if v is None or v0 is None:
            return

        dt = t - t0
        if dt <= 0:
            return 

        vel = (v - v0) / dt
        amp = abs(v - v0)

        if self.debug:
            print(f"[DEBUG] Δv={v - v0:.3f}, Δt={dt:.3f}, vel={vel:.3f}")

        if abs(vel) > self.vel_thresh:
            self.saccades.append((t0, t, amp, vel))
            if self.debug:
                print(f"[SACCADE DETECTED] amp={amp:.3f}, vel={vel:.3f}")
        elif abs(vel) > self.jitter_thresh:
            self.jitters.append((t, vel))
            if self.debug:
                print(f"[JITTER DETECTED] vel={vel:.3f}")

