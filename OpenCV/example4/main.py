import cv2
import numpy as np

# read the image
img = cv2.imread("example4_2.jpg")

# resize the image
img = cv2.resize(img, (500, 500))

# load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# detect faces in the image
faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

cv2.imwrite("result.jpg", img)
print("[MAIN] Image saved successfully!")