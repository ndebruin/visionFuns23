import numpy as np
import cv2

# neural net
net = cv2.dnn.readNetFromTorch('model.pt')

img = cv2.imread("ex.png")

# Capture frame-by-frame
ret, frame = img.read()

# Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Display the resulting frame
cv2.imshow('frame',frame)
cv2.imshow('grayscale',gray)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
img.release()
cv2.destroyAllWindows()
