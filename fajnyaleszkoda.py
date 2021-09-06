#import RPi.GPIO as GPIO
import cv2
import imutils
import numpy as np
import datetime
import threading
from threading import Timer,Thread,Event
import time
import math
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.image as mpframe
import numpy as np
import cv2
import os
import math
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_EXPOSURE, 20)

cam = cv2.VideoCapture(2)


def detect_Line(frame):
    downWhite = np.array([0, 0, 87])
    upWhite = np.array([179, 255, 255])
    blurred = cv2.blur(crop_roi(frame), (10, 100), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, downWhite, upWhite)
    kernel = np.ones((30, 30), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(thresh,0,0)
    invert = cv2.bitwise_not(thresh)
    return invert


def detect_White(frame):
    #downWhite = np.array([27, 66, 105])
    #upWhite = np.array([80, 242, 224])
    downWhite = np.array([0, 0, 87])
    upWhite = np.array([179, 255, 255])
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, downWhite, upWhite)
    kernel = np.ones((30, 30), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closing

def aaa(frame):
    blurred = cv2.blur(frame, (1, 1), 0)
    aaa = blurred[30:30, 30:30]
    return aaa



def region_of_interest(frame, vertices):
    mask = np.zeros_like(frame)
    invert = cv2.bitwise_not(frame)   

    if len(invert.shape) > 2:
        channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_xor(invert, mask)

    return masked_image

def crop_roi(frame):
    mask = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    contours = cv2.findContours(detect_White(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
            cv2.drawContours(frame2, [approx], 0, (255), 5)
            if len(approx) == 4:
                x = approx.ravel()[0]
                y = approx.ravel()[1]
                x2 = approx.ravel()[2]
                y2 = approx.ravel()[3]
                x3 = approx.ravel()[4]
                y3 = approx.ravel()[5]
                x4 = approx.ravel()[6]
                y4 = approx.ravel()[7]

                szeg = int((x2+x3)/2)
                szed = int((x4+x)/2)
                alpas = 0.25
                alkryt = 0.55
                pszegl = int(szeg-(szeg*alpas))
                pszegp = int(szeg+(szeg*alpas))
                pszedl = int(szed-(szed*alpas))
                pszedp = int(szed+(szed*alpas))
                aszegl = int(szeg-(szeg*alkryt))
                aszegp = int(szeg+(szeg*alkryt))
                aszedl = int(szed-(szed*alkryt))
                aszedp = int(szed+(szed*alkryt))
                cv2.line(frame2, (pszedl, y), (pszegl, y2), (0, 255, 255), 2)
                cv2.line(frame2, (pszegp, y3), (pszedp, y4), (0, 255, 255), 2)
                cv2.line(frame2, (aszedl, y), (aszegl, y2), (0, 0, 255), 2)
                cv2.line(frame2, (aszegp, y3), (aszedp, y4), (0, 0, 255), 2)
            else:
                print("ni ma")
    except:
        pass
    
    roi = contours
    return region_of_interest(frame, roi)


def find_pasek(frame):
    contours = cv2.findContours(detect_Line(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pas = []
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        # Obtain bounding rectangle to get measurements
        
        # Find centroid
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        pas.append(cX)
        cv2.circle(frame2, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(frame2, "Pasek", (cX, cY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return pas

while (True):

    ret, frame = cap.read()
    ret, frame2 = cap.read()

    width = 800
    frame = imutils.resize(frame, width)
    frame2 = imutils.resize(frame2, width)

    try:
        crop_roi(frame)
        find_pasek(frame)
        print(find_pasek(frame))
    except:
        pass





    cv2.imshow('Main', frame2)
    cv2.imshow('1', detect_Line(frame))
    cv2.imshow('3', crop_roi(frame))


    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()