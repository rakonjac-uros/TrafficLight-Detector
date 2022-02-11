#!/usr/bin/env python
# coding: utf-8
# created by hevlhayt@foxmail.com 
# Date: 2016/1/15 
# Time: 19:20
#
import os
import cv2
import numpy as np
import argparse


def detect(filepath, file, method):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(filepath+file)
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0,6,252])
    upper_red1 = np.array([35,90,255])
    
    lower_red2 = np.array([0,0,252])
    upper_red2 = np.array([170,40,255])
    
    lower_red3 = np.array([140,0,255])
    upper_red3 = np.array([180,75,255])
    
    lower_red4 = np.array([3,12,255])
    upper_red4 = np.array([40,55,255])
    
    lower_red5 = np.array([0,30,150])
    upper_red5 = np.array([30,130,255])
    
    lower_red6 = np.array([0,0,240])
    upper_red6 = np.array([40,40,255])
    
    lower_red7 = np.array([0,0,240])
    upper_red7 = np.array([40,40,255])
    
    lower_red8 = np.array([0,0,240])
    upper_red8 = np.array([40,40,255])
    
    lower_red9 = np.array([0,0,240])
    upper_red9 = np.array([40,40,255])
    
    lower_red10 = np.array([0,0,240])
    upper_red10= np.array([40,40,255])
    
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    # lower_yellow = np.array([15,100,100])
    # upper_yellow = np.array([35,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
    mask4 = cv2.inRange(hsv, lower_red4, upper_red4)
    
    #mask5 = cv2.inRange(hsv, lower_red5, upper_red5)
    
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    maskr1 = cv2.add(mask1, mask2)
    maskr2 = cv2.add(mask3, mask4)

    maskr = cv2.add(maskr1, maskr2)
    #maskr = cv2.add(maskr12, mask5)
    size = img.shape
    # print size
    
    # Copy the thresholded image.
    im_floodfill = maskr.copy()
    h, w = maskr.shape[:2]
    maskk = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, maskk, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = maskr | im_floodfill_inv
    cv2.imshow('ff', im_out)
    kernel = np.ones((2,2), np.uint8)
    img_dilation = cv2.dilate(maskr, kernel, iterations=1)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    maskrr = maskr.copy()
    maskr = img_erosion   
    # hough circle detect
    if (method == "hg"):
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=5, minRadius=3, maxRadius=14)

        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                     param1=50, param2=10, minRadius=0, maxRadius=30)

        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                     param1=50, param2=5, minRadius=0, maxRadius=30)
    elif (method == "hg_alt"):
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT_ALT, 1.5, 20,
                                   param1=300, param2=0.4, minRadius=3, maxRadius=14)

        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT_ALT, 1.5, 60,
                                     param1=300, param2=10, minRadius=0, maxRadius=30)

        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT_ALT, 1.5, 30,
                                     param1=300, param2=5, minRadius=0, maxRadius=30)

    # traffic light detect
    r = 5
    bound = 0.0 / 10
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0]or i[1] < size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50    :
                cv2.circle(cimg, (i[0], i[1]), i[2]+2, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+2, (255, 255, 255), 2)
                cv2.putText(cimg,'ON',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    numpy_horizontal_concat = np.concatenate((cv2.cvtColor(img_erosion, cv2.COLOR_GRAY2BGR), cimg), axis=1)
    
    #cv2.imwrite(path+'//result//'+file, cimg)
    #cv2.imshow('maskr', maskrr)
    #cv2.imshow('erode', img_erosion)
    #cv2.imshow('dialate', img_dilation)
    #cv2.imshow('detected results', cimg)
    #cv2.imshow('horizontal', numpy_horizontal)
    cv2.imshow('horizontal_c', numpy_horizontal_concat)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--method", help="Hough Gradient (hg or hg_alt)")
    path = os.path.abspath('..')+'//light//'
    args = a.parse_args()
    for f in os.listdir(path):
        print (f)
        if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png') or f.endswith('.PNG'):
            detect(path, f, args.method)

