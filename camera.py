from main import *

import numpy as np 
import cv2

def writeOnImage (frame, hand):
    text = "Searching ..."

    if framesElapsed < CALIBRATION_TIME:
        text = "Calibrating ..."
    elif hand == None or hand.isInFrame == False:
        text = "No Hand detected"
    else:
        if hand.isWaving:
            text = "Waving"
        elif hand.showedFingers == 0:
            text = "Rock"
        elif hand.showedFingers == 1:
            text = "Pointing"
        elif hand.showedFingers == 2:
            text = "Scissors"
    
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.rectangle(frame, (regionLeft, regionTop), (regionRight, regionBottom), (255, 255, 255), 2)

def getRegion (frame):
    region = frame[regionTop:regionBottom, regionLeft:regionRight]
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = cv2.GaussianBlur(region, (5, 5), 0)

    return region 

def getAvarage (region):
    global background

    if background is None:
        background = region.copy().astype("float")
        return

    cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment (region):
    global hand 

    diff = cv2.absdiff(background.astype(np.uint8), region)
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        if hand is not None:
            hand.isInFrame = False

        return
    else:
        if hand is not None:
            hand.isInFrame = True

        segmented_region = max(contours, key = cv2.contourArea)
        return (thresholded_region, segmented_region)

class HandData:
    top = (0, 0)
    bottom = (0, 0)
    left = (0, 0)
    right = (0, 0)
    
    centerX = 0
    prevCenterX = 0

    isInFrame = False
    isWaving = False 

    showedFingers = None
    gestureList = []

    def __init__ (self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
        self.centerX = centerX
        self.prevCenterX = 0

        isInFrame = False
        isWaving = False
    
    def update (self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def isCurrentlyWaving (self, centerX):
        self.prevCenterX = self.centerX
        self.centerX = centerX
        
        if abs(self.centerX - self.prevCenterX > 3):
            self.isWaving = True
        else:
            self.isWaving = False

def countFingers (thresholded_image):
    line_height = int(hand.top[1] + (0.2 * (hand.bottom[1] - hand.top[1])))
    line = np.zeros(thresholded_image.shape[:2], dtype = int)

    cv2.line(line, (thresholded_image.shape[1], line_height), (0, line_height), 255, 1)
    
    line = cv2.bitwise_and(thresholded_image, thresholded_image, mask = line.astype(np.uint8))
    _, contours, _ = cv2.findContours(line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fingers = 0 

    for curr in contours:
        width = len(curr)

        if width < 3 * abs(hand.right[0] - hand.left[0]) / 4 and width > 5:
            fingers += 1

    return fingers

def mostFrequent (input_list):
    dict = {}

    count = 0 
    most_freq = 0 

    for item in reversed (input_list):
        dict[item] = dict.get(item, 0) + 1

        if dict[item] >= count:
            count, most_freq = dict[item], item

    return most_freq

def getHandData(thresholded_image, segmented_image):
    global hand 

    convexHull = cv2.convexHull(segmented_image)

    top = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right = tuple(convexHull[convexHull[:, :, 0].argmax()][0])
    
    centerX = int((left[0] + right[0]) / 2)

    if hand == None:
        hand = HandData(top, bottom, left, right, centerX)
    else:
        hand.update(top, bottom, left, right)

    if framesElapsed % 6 == 0:
        hand.isCurrentlyWaving(centerX)

    hand.gestureList.append(countFingers(thresholded_image))
    
    if framesElapsed % 12 == 0:
        hand.showedFingers = mostFrequent(hand.gestureList)
        hand.gestureList.clear()


capture = cv2.VideoCapture(0)

while (True):
    ret, frame = capture.read()

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)

    region = getRegion(frame)

    if framesElapsed < CALIBRATION_TIME:
        getAvarage(region)
    else:
        region_pair = segment(region)

        if region_pair is not None:
            (thresholded_region, segmented_region) = region_pair

            cv2.drawContours(region, [segmented_region], -1, (255, 255, 255))
            cv2.imshow("Segmented Image", region)    

            getHandData(thresholded_region, segmented_region)

    writeOnImage(frame, hand)
    cv2.imshow("Project: Hand Gesture", frame)

    framesElapsed += 1

    if (cv2.waitKey(1) & 0xFF == ord("x")):
        break

capture.release()
cv2.destroyAllWindows()
