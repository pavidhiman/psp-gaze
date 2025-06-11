"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking, PSPGazeMetrics

gaze = GazeTracking()
metrics = PSPGazeMetrics(gaze, debug=True)
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    snapshot = metrics.update(frame)     
    frame     = gaze.annotated_frame()

    # saccade info
    text = "Blinking"          if gaze.is_blinking() else \
           "Look right"        if gaze.is_right()    else \
           "Look left"         if gaze.is_left()     else \
           "Look centre"

    cv2.putText(frame, text, (90, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # last detected horizontal & vertical saccades 
    if snapshot["last_horiz_saccade"]:
        cv2.putText(frame, "H-saccade!", (90, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
    if snapshot["last_vert_saccade"]:
        cv2.putText(frame, "V-saccade!", (90, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 1)

    cv2.imshow("PSP-Gaze Demo", frame)
    if cv2.waitKey(1) == 27:      # ESC to quit
        break

   
webcam.release()
cv2.destroyAllWindows()
