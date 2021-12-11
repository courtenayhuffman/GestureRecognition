import cv2 as cv
import numpy as np
from keras.models import load_model

CATEGORY_MAP = {
    "call_me": 0,
    "fingers_crossed": 1,
    "okay": 2,
    "paper": 3,
    "peace": 4,
    "rock": 5,
    "rock_on": 6,
    "scissor": 7,
    "thumbs": 8,
    "up": 9
}

def label_mapper(val):
    return CATEGORY_MAP[val]

# CAMERA INTERFACE
def get_webcam():
    cap = cv.VideoCapture(0) #replace with filepath to read video
    #cap.open(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while(True):
        ret, frame = cap.read() # frame captured stored in frame var
        predict_gesture(frame)
        cv.imshow('Input', frame)
        c = cv.waitKey(1)
        if c == 27: # escape key to exit
            break
    cap.release()
    cv.destroyAllWindows

# ML MODEL USE
def predict_gesture(frame):
    img = cv.imread(frame)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (225, 225))
    model = load_model("gesture-model.h5")
    prediction = model.predict(np.array([img]))
    gesture_numeric = np.argmax(prediction[0]) #what is this even doing
    gesture_name = label_mapper(gesture_numeric) 
    print("Predicted Gesture: {}".format(gesture_name))
    return

get_webcam()

# maybe??
# def detect_objects(img, net, outputLayers): # image, model, output layers
# 	blob = cv.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
# 	net.setInput(blob)
# 	outputs = net.forward(outputLayers)
# 	return blob, outputs