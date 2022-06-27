from camera import *

CALIBRATION_TIME = 30 
BG_WEIGHT = 0.5 
OBJ_THRESHOLD = 18

FRAME_HEIGHT = 500
FRAME_WIDTH = 600

background = None
hand = None

framesElapsed = 0

regionTop = 0 
regionBottom = int(2 * FRAME_HEIGHT / 3)
regionLeft = int(FRAME_WIDTH / 2)
regionRight = FRAME_WIDTH
