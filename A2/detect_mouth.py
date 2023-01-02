import cv2
import os


def mouthdetection():
    pic = cv2.imread('../Datasets/celeba/img/0.jpg')
    haarface = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    haarmouth = cv2.CascadeClassifier('haarcascade_mouth.xml')
    face = haarface.detectMultiScale(pic, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
    if len(face) > 0: # face detected
        for rectface in face:
            x, y, w, h = rectface
            intx = int(x)
            inty = int(y)
            intw = int(w)
            inth = int(h)
            mouth1 = int(float(y + 0.7 * h))
            mouth2 = int(0.4 * h)
            halfface_down = pic[mouth1:(mouth1 + mouth2), intx:intx + intw]
            cv2.rectangle(pic, (int(x) ,mouth1), (int(x) + int(w), mouth1 + mouth2), (0, 255, 0), 2, 0)
            mouth = haarmouth.detectMultiScale(halfface_down, 1.1, 1, cv2.CASCADE_SCALE_IMAGE, (5, 20))
            if len(mouth) > 0: #mouth detected
                for rectmouth in mouth:
                    xm, ym, wm, hm = rectmouth
                    cv2.rectangle(halfface_down, (int(xm), int(ym)), (int(xm) + int(wm), int(ym) + int(hm)), (0, 0, 255), 2, 0)



mouthdetection()

