# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 23:21:55 2023

@author: Karolina
"""


    """
    etykiety = sime.label(otwarcie) 
    cechy = sime.regionprops(etykiety)
   

    
    plt.imshow(video_korekta, cmap=plt.cm.gray)
    for cecha in cechy:
        y0, x0 = cecha.centroid 
        plt.plot(x0, y0, '.r', markersize=20) 
        minr, minc, maxr, maxc = cecha.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-y', linewidth=1) 
    #plt.show()

   
    if len(cechy) > 0: 
        for cecha in cechy:
            proporcja = cecha.area_filled/cecha.area_bbox
            minr, minc, maxr, maxc = cecha.bbox
           
            if (proporcja < 0.60):
                centr = tuple((int(cecha.centroid[1]), int(cecha.centroid[0])))
                cv2.putText(video_org, 'Nozyczki', centr,  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
                cv2.rectangle(video_org, (minc, minr), (maxc, maxr), (255,0,0), 1)
            elif (proporcja > 0.60 and proporcja < 0.8):
                centr = tuple((int(cecha.centroid[1]), int(cecha.centroid[0])))
                cv2.putText(video_org, 'Nakretka', centr,  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
                cv2.rectangle(video_org, (minc, minr), (maxc, maxr), (0,255,0), 1)
            elif proporcja > 0.8:
                centr = tuple((int(cecha.centroid[1]), int(cecha.centroid[0])))
                cv2.putText(video_org, 'Paszport', centr,  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
                cv2.rectangle(video_org, (minc, minr), (maxc, maxr), (0,0,255), 1)
"""
