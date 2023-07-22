# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:51:24 2023

@author: nitu
"""

import cv2

if __name__=="__main__":
    video= cv2.VideoCapture(0)
    face_detector= cv2.CascadeClassifier("D:\\Profile_building\\Face_detection and blurring\\haarcascade_frontalface_default.xml")
    
    while True:
        ret,frame= video.read()
        frame_copy= frame.copy()
        if ret== False:
            continue
        frame_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections= face_detector.detectMultiScale(frame_gray, scaleFactor= 1.1)
        frame_copy= cv2.blur(frame,(80,80))
        for (x,y,w,h) in detections:
            #frame= cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0),2)
            frame_copy[y:y+h,x:x+h]= frame[y:y+h,x:x+h]
            
        cv2.imshow('frame', frame_copy)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video.release()
    cv2.destroyAllWindows()
        