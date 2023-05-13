import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    smoothed = cv2.edgePreservingFilter(frame, flags=1, sigma_s=60, sigma_r=0.4)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('smooth',smoothed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()