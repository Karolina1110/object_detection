# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:11:53 2023

@author: Karolina
"""

import cv2
import numpy as np 
import skimage.measure as sime
import matplotlib.pyplot as plt


rozmycie = 7
rozmiar_maski = 5
pokaz_wszystko = False

# ustalenie zrodla wideo
video = cv2.VideoCapture('http://192.168.0.59:8080/video') 


while(True):

    # przechwycenie obrazu oryginalnego
    _, video_org = video.read()
    # zmiana rozmiaru wideo
    video_org = cv2.resize(video_org, (600,400))
    # zmiana obrazu kolorowego na czarno-bialy
    video_szare = cv2.cvtColor(video_org, cv2.COLOR_BGR2GRAY)
    # rozmycie w celu usuniecia szumu
    video_blur = cv2.GaussianBlur(video_szare,(rozmiar_maski,rozmiar_maski),rozmycie*2) 
    # binaryzacja wideo
    _,video_binary = cv2.threshold(video_blur,120, 255, cv2.THRESH_BINARY) 
    # filtr otwarcia
    elstr = np.ones((3,3),np.uint8)
    otwarcie = cv2.morphologyEx(video_binary,cv2.MORPH_OPEN,elstr)

    # wszystkie parametry detekcji klasyfikują obiekt na podstawie jego okraglosci

    # parametry do klasyfikacji nakretki
    par1 = cv2.SimpleBlobDetector_Params()
    par1.minThreshold = 120
    par1.maxThreshold = 255
    par1.filterByArea = False
    par1.filterByCircularity = True
    par1.minCircularity=0.75
    par1.maxCircularity=1.0
    par1.filterByConvexity=False
    par1.filterByInertia=False
    # detekcja
    detektor1 = cv2.SimpleBlobDetector_create(par1)
    obiekty1 = detektor1.detect(otwarcie)
    # zaznaczenie
    zaznaczenie1 = cv2.drawKeypoints(video_org, obiekty1, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # podpis
    for obj in obiekty1:
        centr = tuple((int(obj.pt[0]), int(obj.pt[1])))
        cv2.putText(zaznaczenie1, 'Nakretka', centr,  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
    
    # parametry do klasyfikacji paszportu
    par2 = cv2.SimpleBlobDetector_Params()
    par2.minThreshold = 120
    par2.maxThreshold = 255
    par2.filterByArea = False
    par2.filterByCircularity = True
    par2.minCircularity=0.55
    par2.maxCircularity=0.74
    par2.filterByConvexity=False
    par2.filterByInertia=False
    # detekcja
    detektor2 = cv2.SimpleBlobDetector_create(par2)
    obiekty2 = detektor2.detect(otwarcie)
    # zaznaczenie
    zaznaczenie2 = cv2.drawKeypoints(video_org, obiekty2, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # podpis
    for obj in obiekty2:
        centr = tuple((int(obj.pt[0]), int(obj.pt[1])))
        cv2.putText(zaznaczenie2, 'Paszport', centr,  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
    
    # parametry do klasyfikacji srubki
    par3 = cv2.SimpleBlobDetector_Params()
    par3.minThreshold = 120
    par3.maxThreshold = 255
    par3.filterByArea = False
    par3.filterByCircularity = True
    par3.minCircularity=0.0
    par3.maxCircularity=0.54
    par3.filterByConvexity=False
    par3.filterByInertia=False
    # detekcja
    detektor3 = cv2.SimpleBlobDetector_create(par3)
    obiekty3 = detektor3.detect(otwarcie)
    # zaznaczenie
    zaznaczenie3 = cv2.drawKeypoints(video_org, obiekty3, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # podpis
    for obj in obiekty3:
        centr = tuple((int(obj.pt[0]), int(obj.pt[1])))
        cv2.putText(zaznaczenie3, 'Srubka', centr,  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
    
    # nalozenie obrazow z trzech detekcji na siebie
    zaznaczenie = cv2.addWeighted(zaznaczenie1, 0.5, zaznaczenie2, 0.5, 0.0)
    zaznaczenie = cv2.addWeighted(zaznaczenie, 0.5, zaznaczenie3, 0.5, 0.0)

    # wizualizacja wynikow
    if pokaz_wszystko:
        
        cv2.imshow("obraz monochromatyczny", video_szare)
        cv2.imshow("obraz rozmyty", video_blur)
        cv2.imshow("obraz binarny", video_binary)
    cv2.imshow("obraz otwarcie", otwarcie)
    cv2.imshow("obraz oryginalny", video_org)
    cv2.imshow("obraz z zaznaczeniem", zaznaczenie)
    
    k = cv2.waitKey(1) & 0xFF 
    if k == 27:
        break
video.release()
cv2.destroyAllWindows()



