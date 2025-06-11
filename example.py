import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.psp_metrics import PSPGazeMetrics
from gaze_tracking.logger import EventLogger

gaze   = GazeTracking()
logger = EventLogger("session_1.csv")
metrics = PSPGazeMetrics(gaze, logger=logger, debug=True)   # set debug=False to stop console prints
webcam  = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        snapshot = metrics.update(frame)          
        frame    = gaze.annotated_frame()         

        # simple gaze-direction text
        text = ("Blinking"   if gaze.is_blinking() else
                "Look right" if gaze.is_right()    else
                "Look left"  if gaze.is_left()     else
                "Look centre")

        cv2.putText(frame, text, (90, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        # overlay last detected saccades
        if snapshot["last_horiz_saccade"]:
            cv2.putText(frame, "H-saccade!", (90, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
        if snapshot["last_vert_saccade"]:
            cv2.putText(frame, "V-saccade!", (90, 130),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 1)

        cv2.imshow("PSP-Gaze Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

except KeyboardInterrupt:
    pass # used for CTRL + C

finally:
    logger.to_csv()                     
    webcam.release()
    cv2.destroyAllWindows()
