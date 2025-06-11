import csv
import time
from pathlib import Path

class EventLogger:
    # logs data into csv 
    def __init__(self, write_file="gaze_log.csv"):
        self.rows = [] # buffer 
        self.start = time.time()
        self.path = Path(write_file)
        
    # functions called by PSPGazeMetrics
    def log_frame(self, t, h, v, blink):
        self.rows.append((t, "FRAME",
                          f"h={h:.3f}" if h is not None else "h=None",
                          f"v={v:.3f}" if v is not None else "v=None",
                          f"blink={blink}"))

    def log_event(self, t0, t1, amp, vel, axis, kind):
        self.rows.append((t1, f"{axis}-{kind}",
                          f"amp={amp:.3f}",
                          f"vel={vel:.3f}",
                          f"dt={t1-t0:.3f}"))
        
    # exporting
    def to_csv(self):
        header = ["timestamp", "type", "field1", "field2", "field3"]
        with self.path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.rows)
        print(f"[LOGGER] wrote {len(self.rows)} rows → {self.path.absolute()}")