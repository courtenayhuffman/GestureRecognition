# COGS 300 Term Project
# Gesture Recognition with ML

# things I need to import:
#   tensorflow, keras? pytorch?
#   dataset
#   gulpIO
#   yolo or squeezenet or other
import pandas as pd
import numpy as np
import cv2 as cv
# download and import GulpIO

def get_webcam():
    cap = cv.VideoCapture(0) #replace with filepath to read video
    #cap.open(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while(True):
        ret, frame = cap.read() # frame captured stored in frame var
        cv.imshow('Input', frame)
        c = cv.waitKey(1)
        if c == 27: # escape key to exit
            break
    cap.release()
    cv.destroyAllWindows

get_webcam()
# load dataset
#ds = pd.read('')
# data processing

# NN model
# NN train model

# NN validate model

# NN use model

