import cv2
import csv 
from gaze_tracking import GazeTracking
from gaze_tracking.calibration import Calibration
from gaze_tracking.psp_metrics import PSPGazeMetrics

calib = Calibration()
calib.run()
gaze = GazeTracking()
psp = PSPGazeMetrics(gaze, calib)

cap = cv2.VideoCapture(0)
csvfile = open('psp_data.csv', 'w', newline = '')
writer = csv.writer(csvfile)
writer.writerow(['t','h_ratio','v_ratio','blink','sac_start','sac_end','sac_amp','sac_vel','jitter_t','jitter_vel'])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = psp.update(frame)

        sac = data['saccade'] or (None,)*4
        jit = data['jitter'] or (None,)*2
        writer.writerow([
            data['timestamp'],
            data['h_ratio'],
            data['v_ratio'],
            data['blink'],
            sac[0], sac[1], sac[2], sac[3],
            jit[0], jit[1]
        ])
        csvfile.flush()

        out = gaze.annotated_frame()
        cv2.putText(out, f"V-sac: {sac[2]:.2f}/{sac[3]:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("PSP Gaze", out)
        if cv2.waitKey(1)==27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    csvfile.close()
