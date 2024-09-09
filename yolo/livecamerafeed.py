from PIL import Image
import cv2
from yolo import Yolo
from time import sleep
import torch

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
yolo = Yolo()

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    frame = Image.fromarray(frame)
    frame = yolo.run(frame, bb_size=1)
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()