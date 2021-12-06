import cv2 as cv

# CAMERA INTERFACE
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
