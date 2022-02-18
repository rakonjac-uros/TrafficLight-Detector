#!/usr/bin/env python3
# coding: utf-8


import cv2 as cv
import numpy as np
# Read image given by user

image = cv.imread('/home/uros/Projects/light_detection_testing/TrafficLight-Detector/light/frame69.png')
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
image = hsv
new_image = np.zeros(image.shape, image.dtype)
alpha = 2 # Simple contrast control
beta = 10    # Simple brightness control
gamma = 10
# Initialize values
print(' Basic Linear Transforms ')
print('-------------------------')

# Do the operation new_image(i,j) = alpha*image(i,j) + beta
# Instead of these 'for' loops we could have used simply:
# new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# but we wanted to show you how to access the pixels :)
#for y in range(image.shape[0]):
#    for x in range(image.shape[1]):
#        for c in range(image.shape[2]):
#            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

#numpy_horizontal_concat = np.concatenate((new_image, image), axis=0)

#cv.imshow('Averaging',numpy_horizontal_concat)
#cv.waitKey(0)

for g in range (20):
    gamma = 2*g
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv.LUT(image, lookUpTable)
    
    cv.imwrite('/home/uros/Projects/light_detection_testing/TrafficLight-Detector/light/result/'+str(gamma)+'.png', res)

#numpy_horizontal_concat = np.concatenate((res, image), axis=0)
#
#cv.imshow('gamma',numpy_horizontal_concat)
#cv.waitKey(0)
#
#
#cv.destroyAllWindows()
