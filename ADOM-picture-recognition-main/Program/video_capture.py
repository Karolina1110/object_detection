# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:31:57 2022

@author: Olimpia
"""

import cv2
import numpy as np 

video = cv2.VideoCapture(0) #function which capture videao strem, paramiter = 0,1,2 is a camera number, output= obiekt video

low_threshold = np.array([100,0,0])
upper_threshold = np.array([255,200,155])

while(True):
    ret, original_img = video.read() # output: array original_img, ret = 0 or False -> we cannot read original_img
    
    blurry_img = cv2.GaussianBlur(original_img,(5,5),4)
    mask_img = cv2.inRange(original_img, low_threshold, upper_threshold) # iRange is a binarization operation in which we specify a range of color component values that correspond to "low_threshold", "upper_threshold"
    
    
    # cv2.imshow displays the image in a new window, so that the real-time image is always displayed in the same window, assign a window name to the display object.
    cv2.imshow('original_img', original_img)
    
    cv2.imshow('blurry_img', blurry_img)
    cv2.imshow('mask_img', mask_img)
    
    k = cv2.waitKey(1) & 0xFF #press ESC to finish video capture
    if k == 27:
        break
video.release() #freeing up the memory buffer needed to display the image
cv2.destroyAllWindows()# close windows which video display