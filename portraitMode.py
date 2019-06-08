import cv2
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import math

img = cv2.imread('images/test3.jpg',cv2.COLOR_BGR2RGB)
back = img.copy()
blurred = cv2.GaussianBlur(img, (51,51),0)


rects = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml').detectMultiScale(img,1.3, 5)

for f in rects:
    x, y, w, h = [ v for v in f ]
    r = (2*x+w)/2

    sub_face = img[y:y+h+10, x:x+w+10]
    blurred[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

cv2.imwrite('outputs/test3.jpg',blurred)
