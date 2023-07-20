# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:29:39 2023

@author: nitu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
     
    image= cv2.imread("D:\\Profile_building\\Face_detection and blurring\\people2.jpg")
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #image= cv2.resize(image,(800,600))
    print(image.shape)
    
    image_gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector= cv2.CascadeClassifier("D:\Profile_building\Face_detection and blurring\haarcascade_frontalface_default.xml")
    detections= face_detector.detectMultiScale(image_gray, scaleFactor= 1.02, minNeighbors=7)
    
    for (x,y,w,h) in detections:
        #cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 3)
        image_blur= cv2.blur(image[y:y+h, x:x+w], (15,15))
        image[y:y+h, x:x+w]= image_blur
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    filename= 'people2_blurred.jpg'
    
    cv2.imwrite(filename, image)
    