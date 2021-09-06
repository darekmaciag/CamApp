import cv2
import imutils
import numpy as np
import datetime
import threading
from threading import Timer,Thread,Event
import time
import threading


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.ruch = 0
        self.greenLed = 7
        self.redLed = 5
        self.enablePin = 17 # Note enable is active low!
        self.directionPin = 24
        self.pulsePin = 23
        self.alarmPin = 20
        self.buttonPin = 13
        self.zasilaczPin = 21
        self.speed = 0.0001
        thread = threading.Thread(target=self.move)
        thread.daemon = True
        thread.start()


    def setup(self):
        # define  GPIOs

        # initialize GPIO
        print("\n Starting..... ==> Press 'q' to quit Program \n")
        ledOn = False
        alarm = False

    def move(self):
        while True:
            if self.ruch == 1:
                print("move")
            else:
                pass


    def detect_white(frame):
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


    def find_line(frame):
        #colorLower = np.array([160, 120, 100])
        #colorUpper = np.array([179, 184, 204])
        colorLower = np.array([0, 120, 100])
        colorUpper = np.array([20, 255, 255])
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        srodek = cv2.inRange(hsv, colorLower, colorUpper)
        kernel = np.ones((20, 3), np.uint8)
        dilation = cv2.dilate(srodek, kernel, iterations=1)
        return dilation


    def startowy(self):
        while True:
            ret, frame = self.cap.read()
            width = 1000
            frame = imutils.resize(frame, width)
            button = 0
    
            contours = cv2.findContours(Camera.detect_white(frame).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
            contours2 = cv2.findContours(Camera.find_line(frame).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

            if len(contours) > 0 and button == 0:
                
                maxcontour = max(contours, key=cv2.contourArea)
                (xg, yg, wg, hg) = cv2.boundingRect(maxcontour)
                a = int(xg + (wg / 2))
                b = int(yg + (hg / 2))
                przes = int(wg / 25)
                alarm = int(wg / 3.2)
                lewo = int(a - przes)
                prawo = int(a + przes)
                alarml = int(a - alarm)
                alarmp = int(a + alarm)
                cv2.circle(frame, (a, b), 5, (0, 0, 0), -1)
                mask = cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)
                cv2.line(frame, (lewo, yg), (lewo, yg + hg), (0, 255, 255), 2)
                cv2.line(frame, (prawo, yg), (prawo, yg + hg), (0, 255, 255), 2)
                cv2.line(frame, (alarml, yg), (alarml, yg + hg), (0, 0, 255), 2)
                cv2.line(frame, (alarmp, yg), (alarmp, yg + hg), (0, 0, 255), 2)
                if len(contours2) > 0:
                    ledOn = True
                    print("led ON")

                    pasek = max(contours2, key=cv2.contourArea)
                    # compute the center of the contour
                    M = cv2.moments(pasek)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cX2 = int(M["m10"] / M["m00"])
                    # draw the contour and center of the shape on the image
                    cv2.drawContours(frame, [pasek], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cX, cY), 1, (255, 255, 255), -1)
                    cv2.putText(frame, "pasek", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if not ledOn:
                        # turn the LED on
                        ledOn = True
                        print("led ON")

                    if cX > prawo:
                        self.ruch = 1
                        print("prawo")

                    elif cX < lewo:
                        self.ruch = 1
                        print("lewo")


                    else:
                        ruch = False
                        self.ruch = 0


                    if cX2 < alarml:
                        alarm = True
                        print("alarm lewo")

                    elif cX2 > alarmp:
                        alarm = True
                        print("alarm prawo")

                    else:
                        alarm = False
                        print("alarm wyl")
                        
                else:
                    ledOn = False
                    ruch = False
                    self.ruch = 0


            else:
                alarm = False
                ledOn = False
                lewo = False
                prawo = False
                self.ruch = 0

            cv2.imshow('1', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n Exiting Program and cleanup stuff \n")        
                #clean without enablePin
                Camera().cap.release()
                cv2.destroyAllWindows()
                break



def main():
    p1 = Camera()
    p1.setup()
    p1.startowy()


if __name__ == '__main__':
    main()



