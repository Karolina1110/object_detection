# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:40:02 2023

@author: Olimpia
"""

import cv2
import numpy as np

GREY_IMAGE = True
COLOR_IMAGE = True
DETECTOR_SENSITIVITY = 30
upper_threshold = np.array([255,255,255])
low_threshold = np.array([DETECTOR_SENSITIVITY,DETECTOR_SENSITIVITY,DETECTOR_SENSITIVITY])

video = cv2.VideoCapture(0)
_ , actual_color_img = video.read()

if GREY_IMAGE:
    previous_grey_img = cv2.cvtColor(actual_color_img, cv2.COLOR_BGR2GRAY) 
if COLOR_IMAGE:
    previous_color_img = actual_color_img
    
def process_image(actual_img,previous_img,type_img):
    diffrence = cv2.absdiff(actual_img,previous_img)
    if type_img == "color":
        mask = cv2.inRange(diffrence,low_threshold,upper_threshold)
    elif type_img == "grey":
       _,mask = cv2.threshold(diffrence,DETECTOR_SENSITIVITY, 255, cv2.THRESH_BINARY)
    changes = (mask > 0).sum()
    print(f'Changes for {type_img} image is {changes}')
    # cv2.imshow displays the image in a new window, so that the real-time image is always displayed in the same window, assign a window name to the display object.
    cv2.imshow(f'original_img_{type_img}', actual_img)
    cv2.imshow(f'blurry_img_{type_img}', diffrence)
    cv2.imshow(f'mask_img_{type_img}', mask)
        
    
while(True):
    _, actual_color_img = video.read()
    if GREY_IMAGE:
        actual_grey_img = cv2.cvtColor(actual_color_img, cv2.COLOR_BGR2GRAY) #convert to grey image
        process_image(actual_grey_img, previous_grey_img,"grey")
        previous_grey_img = actual_grey_img
    if COLOR_IMAGE:
        process_image(actual_color_img, previous_color_img,"color")
        previous_color_img = actual_color_img
    
    k = cv2.waitKey(1) & 0xFF #press ESC to finish video capture
    if k == 27:
        break
video.release() #freeing up the memory buffer needed to display the image
cv2.destroyAllWindows()# close windows which video display