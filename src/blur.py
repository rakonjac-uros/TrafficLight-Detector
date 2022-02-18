#!/usr/bin/env python3
# coding: utf-8

import cv2
import numpy as np

# bat.jpg is the batman image.
img = cv2.imread('/home/uros/Projects/light_detection_testing/TrafficLight-Detector/light/frame69.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = hsv
# make sure that you have saved it in the same folder
# Averaging
# You can change the kernel size as you want
avging = cv2.blur(img,(3,3))
numpy_horizontal_concat = np.concatenate((avging, img), axis=1)

cv2.imshow('Averaging',numpy_horizontal_concat)
cv2.waitKey(0)
  
# Gaussian Blurring
# Again, you can change the kernel size
gausBlur = cv2.GaussianBlur(img, (3,3),0) 
numpy_horizontal_concat = np.concatenate((gausBlur, img), axis=1)
cv2.imshow('Gaussian Blurring', numpy_horizontal_concat)
cv2.waitKey(0)
  
# Median blurring
medBlur = cv2.medianBlur(img,3)
numpy_horizontal_concat = np.concatenate((medBlur, img), axis=1)
cv2.imshow('Media Blurring', numpy_horizontal_concat)
cv2.waitKey(0)
  
# Bilateral Filtering
bilFilter = cv2.bilateralFilter(img,9,75,75)
numpy_horizontal_concat = np.concatenate((bilFilter, img), axis=1)
cv2.imshow('Bilateral Filtering', numpy_horizontal_concat)
cv2.waitKey(0)
cv2.destroyAllWindows()