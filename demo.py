import os
import cv2
import numpy as np


PATH_TO_FILE = input('Enter the file path for video:')
#cap = cv2.VideoCapture('./CarsDrivingUnderBridge.mp4')
cap = cv2.VideoCapture(PATH_TO_FILE)

try:
    while cap.isOpened():
        ret,frame = cap.read()
        if ret == False:
            break
        #print(type(frame))
        cv2.imshow('detected frame', frame)
        k = cv2.waitKey(30) & 0xff
        if  (k == ord('q')) :
            break
    
    cv2.destroyAllWindows()
    cap.release()
            

except Exception as e:
    print(e)
    cap.release()        
