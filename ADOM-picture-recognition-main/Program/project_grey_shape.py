# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:58:21 2023

@author: Olimpia
"""

import cv2
import numpy as np 
import skimage.measure as sime
import matplotlib.pyplot as plt


LOW_THRESHOLD = np.array([100,0,0])
UPPER_THRESHOLD = np.array([255,200,155])
BLURR_PAR = 15
MASK_THRESHOLD_MIN = 100
MASK_THRESHOLD_MAX = 255
MASK_TYPE = cv2.THRESH_BINARY_INV
FEATURE_MARKER_SIZE = 20
TRIANGE_RATIO = 0.7 #the approximate ratio of the area of a triangle to the area of a rectangular area
WHELL_RATIO = 0.8 #the approximate ratio of the area of a wheel to the area of a rectangular area
RECTANGLE_RATIO = 0.7 #the approximate ratio of the area of a rectangle to the area of a rectangular area
DISPLAY_TEST_MODE = True # If True display only mask and plot,else display: 1) original picture 2) grey_picture 3)mask picture 4)blur picture 5) plot


video = cv2.VideoCapture(0) #function which capture videao strem, paramiter = 0,1,2 is a camera number, output= obiekt video


def final_description(img, obj_name, obj_features):
    
    obj_features = tuple((int(obj_features.centroid[1]), int(obj_features.centroid[0])))
    font_type = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    font_color = (0, 0, 0)
    
    return cv2.putText(img, obj_name, obj_features,  font_type, font_size, font_color)


while(True):

    # 1 PICTURE PROCESSING ###########################################

    ## 1.1 Image capture from video
    ret, original_img = video.read() # output: array original_img, ret = 0 or False -> we cannot read original_img
    grey_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) #change color image to grey one for easier recognition

    ## 1.2 White noise 
    blurry_img = cv2.GaussianBlur(grey_img,(5,5),BLURR_PAR) #blurry image to delete white noise
    _,mask_img = cv2.threshold(blurry_img,MASK_THRESHOLD_MIN, MASK_THRESHOLD_MAX, MASK_TYPE) #convert grey image to binary one


    #2 SHAPE RECOGNITION ##############################################
    
    ## 2.1 Mask labeling
    labels = sime.label(mask_img) # return the number of assigned labels

    ##2.2 Assigning features - indicating potential objects in the image
    features = sime.regionprops(labels) #It is some kind of list. feature[0] gives you all features for 1st object, but you need to call with with method like feature[0].centroid()
    obj_count = len(features) #number of features
    plt.imshow(labels, cmap=plt.cm.gray) #this part show objects in yellow frame in the plot
    for obj in features:
        y0, x0 = obj.centroid #centroid = red dot
        plt.plot(x0, y0, '.r', markersize=FEATURE_MARKER_SIZE) #size of red dot
        minr, minc, maxr, maxc = obj.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-y', linewidth=1) #draw plot

    ##2.3 Assign class to features

    if len(features) > 0:  #Sometimes it's need tim to collect alll features. 
        for obj_counter in range(obj_count):
            obj_features = features[obj_counter]
            object_surface = obj_features.area_filled
            object_background_surface = obj_features.area_bbox
            # Conditions for cases : rectangle, oval, squere: - indiwidual part of modification
            if object_surface/object_background_surface < TRIANGE_RATIO: #Then we have triangle
                final_description(mask_img, 'triangle', obj_features)
            elif object_surface/object_background_surface < WHELL_RATIO:
                final_description(mask_img, 'wheel', obj_features)
            else:
                final_description(mask_img, 'rectangle', obj_features) 


    # 3 PICTURE SHOW #################################################

    #cv2.imshow displays the image in a new window, so that the real-time image is always displayed in the same window, assign a window name to the display object.
    if not DISPLAY_TEST_MODE:
        cv2.imshow('original_img', original_img)
        cv2.imshow('grey_img', grey_img)
        cv2.imshow('blurry_img', blurry_img)
    cv2.imshow('mask_img', mask_img)


    # 4 KEYBOARD KEY INSTRUCTIONS ####################################

    k = cv2.waitKey(1) & 0xFF #press ESC to finish video capture
    if k == 27:
        break
video.release() #freeing up the memory buffer needed to display the image
cv2.destroyAllWindows()# close windows which video display



