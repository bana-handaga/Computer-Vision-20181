import numpy as np
import cv2

import time

lbpface_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')


eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('test6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Start Detection
t1 = time.time()
faces = lbpface_cascade.detectMultiScale(gray, 1.2, 5)
# End Detection
t2 = time.time()

print( t2 - t1 )

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()