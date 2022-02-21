#!/usr/bin/env python3
# coding: utf-8
# created by hevlhayt@foxmail.com 
# Date: 2016/1/15 
# Time: 19:20
#
import os
import cv2
from cv2 import CirclesGridFinderParameters
import numpy as np
import argparse
import time

class circle_line:
    def __init__(self, id=0, line=[], top= 0, bot=0, left= 0, right=0, num_on=0):
        self.id = id
        self.top = top
        self.bot = bot
        self.left = left
        self.right = right
        self.line = line
        self.num_on = num_on


def floodfill(mask):
    # Floodfill
    im_floodfill = mask.copy()
    h, w = mask.shape[:2]
    maskk = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, maskk, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = mask | im_floodfill_inv
    return im_out

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]
	return crop_img

def gamma_correction(img, g):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, g) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)


def detect(filepath, file, method = 'hg', gamma = 30, mask_low = np.array([0,0,100]), mask_high = np.array([0,0,255]), dialation_size = (2,2), erosion_size = (2,2), crop_center = True , cc_w = 50, cc_h = 50, draw_circles = False, show_images = False):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img_full = cv2.imread(filepath+file)

    t2 = time.time()

    if (crop_center):
        # Crop image center
        
        center = (img_full.shape[1] / int(100/cc_w), img_full.shape[0] / int(100/cc_h)) 
        img = center_crop(img_full, center)
    else:
        img = img_full

    cimg = img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv = gamma_correction(hsv,gamma)

    # color range
    #lower_red1 = np.array([0,6,252])
    #upper_red1 = np.array([35,90,255])
    #
    #lower_red2 = np.array([0,0,252])
    #upper_red2 = np.array([170,40,255])
    #
    #lower_red3 = np.array([140,0,255])
    #upper_red3 = np.array([180,75,255])
    #
    #lower_red4 = np.array([3,12,255])
    #upper_red4 = np.array([40,55,255])
    
    lower_red5 = mask_low
    upper_red5 = mask_high
    
    #lower_red6 = np.array([0,0,240])
    #upper_red6 = np.array([40,40,255])
    
    #mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    #
    #mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
    #mask4 = cv2.inRange(hsv, lower_red4, upper_red4)
    
    #mask5 = cv2.inRange(hsv, lower_red5, upper_red5)
    
    #maskr1 = cv2.add(mask1, mask2)
    #maskr2 = cv2.add(mask3, mask4)

    #maskr = cv2.add(maskr1, maskr2)

    maskr = cv2.inRange(hsv, lower_red5, upper_red5)
    size = img.shape
    
    # Floodfill
    #im_out = floodfill(maskr)
    #cv2.imshow('ff', im_out)
    
    # Dialation
    kernel = np.ones(dialation_size, np.uint8)
    img_dilation = cv2.dilate(maskr, kernel, iterations=1)

    # Erosion
    kernel = np.ones(erosion_size, np.uint8)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=2)
    
    #maskrr = maskr.copy()
    processed_image = img_erosion   
    
    # hough circle detect
    if (method == "hg"):
        r_circles = cv2.HoughCircles(processed_image, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=4, minRadius=2, maxRadius=14)

    elif (method == "hg_alt"):
        r_circles = cv2.HoughCircles(processed_image, cv2.HOUGH_GRADIENT_ALT, 1.5, 20,
                                   param1=300, param2=0.4, minRadius=3, maxRadius=14)

    print("Detect circles from loaded image; Time: ", time.time()-t2)
    # light detect
    r = 5
    bound = 0.0 / 10
    detected_circles = []
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
                    h += processed_image[i[1]+m, i[0]+n]
                    s += 1
            if (h / s > 50):
                detected_circles.append((i[0], i[1], i[2])) 
                if (draw_circles):
                    cv2.circle(cimg, (i[0], i[1]), i[2]+2, (0, 255, 0), 2)
                    cv2.circle(maskr, (i[0], i[1]), i[2]+2, (255, 255, 255), 2)
                    #cv2.putText(cimg,'ON',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    print ("Detect lights from loaded image; Time: ", time.time() - t2)

    radius_diff_ratio = 1
    circle_lines = []
    processed = []
    for idx,c in enumerate(detected_circles):
        if idx in processed:
            continue
        processed.append(idx)
        curr_line = circle_line(idx, [c], c[1]-c[2], c[1]+c[2], c[0] - c[2], c[0] + c[2], 1)
        for idx2,c2 in enumerate(detected_circles):
            if idx2 in processed:
                continue
            if (abs((curr_line.bot + curr_line.top)/2 - c2[1]) < 2*c[2]): # and (abs((curr_line.bot - curr_line.top)/2 - c2[2]) < c2[2] * radius_diff_ratio):
                processed.append(idx2)
                curr_line.line.append(c2)
                curr_line.num_on += 1
                if (c2[0] + c2[2] > curr_line.right):
                    curr_line.right = c2[0] + c2[2]
                elif (c2[0] - c2[2] < curr_line.left):
                    curr_line.left = c2[0] - c2[2]   
                 
                if (c2[1] - c2[2] < curr_line.top):
                    curr_line.top = c2[1] - c2[2]
                elif (c2[1] + c2[2] > curr_line.bot): 
                    curr_line.bot = c2[1] + c2[2]
        circle_lines.append(curr_line)
        cv2.rectangle(cimg,(curr_line.left, curr_line.top),(curr_line.right, curr_line.bot), (255,0,0), 2)
        cv2.putText(cimg, str(curr_line.num_on) + 'ON', (curr_line.left,curr_line.top-5), font, 1, (255,0,0),2,cv2.LINE_AA)

        
    if (show_images):
        numpy_horizontal_concat = np.concatenate((cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR), cimg), axis=1)
        numpy_horizontal_concat_mh = np.concatenate((cv2.cvtColor(maskr, cv2.COLOR_GRAY2BGR), hsv), axis=1)
        numpy_horizontal_concat_ed = np.concatenate((img_dilation, img_erosion), axis=1)
        

        cv2.namedWindow('mask + hsv')        # Create a named window
        cv2.moveWindow('mask + hsv', 40,30)

        cv2.namedWindow('dialate + erode')        # Create a named window
        cv2.moveWindow('dialate + erode', 40,400)

        cv2.namedWindow('proccesed + final')        # Create a named window
        cv2.moveWindow('proccesed + final', 40,800)
        #cv2.imwrite(path+'//result//'+file, cimg)
        #cv2.imshow('maskr', maskr)
        #cv2.imshow('erode', img_erosion)
        #cv2.imshow('dialate', img_dilation)
        cv2.imshow('mask + hsv', numpy_horizontal_concat_mh)
        cv2.imshow('dialate + erode', numpy_horizontal_concat_ed)
        #cv2.imshow('detected results', cimg)
        #cv2.imshow('horizontal', numpy_horizontal)
        cv2.imshow('proccesed + final', numpy_horizontal_concat)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return detected_circles

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--method", help="Hough Gradient (hg or hg_alt)")
    path = os.path.abspath('..')+'//light//'
    args = a.parse_args()
    for f in os.listdir(path):
        print (f)
        if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png') or f.endswith('.PNG'):
            t1 = time.time()
            lights = detect(path, f, method = 'hg', gamma = 30, mask_low = np.array([0,0,100]), mask_high = np.array([0,0,255]), \
            dialation_size = (2,2), erosion_size = (2,2), crop_center = True , cc_w = 50, cc_h = 50, draw_circles = True, show_images = True)
            
            print ("Total detect time: ",time.time() - t1)

