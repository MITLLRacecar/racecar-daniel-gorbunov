"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 5 - AR Markers
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from enum import IntEnum
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here

### MARKERS ###
BLUE_MARKER = ((90, 100, 100), (120, 255, 255), "blue")
RED_MARKER = ((170, 100, 100), (10, 255, 255), "red")

### WALL FOLLOWING ###
DRIVE_SPEED = 1.0
TURN_ANGLE = 0.3
LEFT_WINDOW = (-50, -40) # center : -45
RIGHT_WINDOW = (40, 50) # center : 45

### LINE FOLLOWING ###
BLUE = ((90, 50, 50), (120, 255, 255))
RED = ((0, 50, 50), (20, 255, 255))
GREEN = ((60, 50, 50), (80, 255, 200))
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
MIN_CONTOUR_AREA = 30

speed, angle = 0, 0
colorImage, depthImage, lidarScan = None, None, None

class State(IntEnum) :
    turnLeft = 0
    turnRight = 1
    followLine = 2
    followWall = 3

robotState = State.followWall

########################################################################################
# Functions
########################################################################################

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Lab 5 - AR Markers")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed, angle, colorImage, depthImage, lidarScan
    colorImage = rc.camera.get_color_image()
    depthImage = rc.camera.get_depth_image()
    lidarScan = rc.lidar.get_samples()

    markers = rc_utils.get_ar_markers(colorImage)
    rc_utils.draw_ar_markers(colorImage, markers)
    
    if len(markers) != 0 : 

        for marker in markers :
            marker.detect_colors(colorImage, [BLUE_MARKER, RED_MARKER])

        # TODO : Marker sorting by distance
        marker = markers[0]
        # TODO: Turn at a certain distance
        if marker.get_id() == 0 :
            robot_state = State.turnLeft
        elif marker.get_id() == 1 :
            robot_state = State.turnRight
        elif marker.get_id() == 2:
            robot_state = State.followLine
        elif marker.get_id() == 199:
            if marker.get_orientation() == marker.get_orientation().LEFT :
                robot_state = State.turnLeft
            elif marker.get_orientation() == marker.get_orientation().RIGHT:
                robot_state = State.turnRight
        else :
            robot_state = State.followWall  

    else : 
        robot_state = State.followWall

    if robot_state == State.followWall :
        followWall()
    elif robot_state == State.followLine :
        followLine(marker.get_color())
    elif robot_state == State.turnLeft :
        turn(-TURN_ANGLE)
    else :
        turn(TURN_ANGLE)

    # If we see a marker with ID 2, follow the color line which matches the color
    # border surrounding the marker (either blue or red). If neither color is found but
    # we see a green line, follow that instead.

    speed = DRIVE_SPEED
    rc.drive.set_speed_angle(speed, angle)
    rc.display.show_color_image(colorImage)

def followLine(color: str): 
    global angle, colorImage
    # Crop the image to the floor directly in front of the car
    croppedImage = rc_utils.crop(colorImage, CROP_FLOOR[0], CROP_FLOOR[1])

    # Find all of the contours
    contours = [rc_utils.find_contours(croppedImage, RED[0], RED[1]), 
        rc_utils.find_contours(croppedImage, GREEN[0], GREEN[1]),
        rc_utils.find_contours(croppedImage, BLUE[0], BLUE[1])]

    # Select the largest contours
    largest_contours = [rc_utils.get_largest_contour(contours[0], MIN_CONTOUR_AREA),
        rc_utils.get_largest_contour(contours[1], MIN_CONTOUR_AREA),
        rc_utils.get_largest_contour(contours[2], MIN_CONTOUR_AREA)]

    if color == "red" :
        contour = largest_contours[0]
    elif color == "blue" :
        contour = largest_contours[2]
    else :
        contour = largest_contours[1]

    contour_center = rc_utils.get_contour_center(contour)

    # Draw contour onto the image
    rc_utils.draw_contour(croppedImage, contour)
    rc_utils.draw_circle(croppedImage, contour_center)

    # scale angle bounds to that of camera
    scale = 1 / (rc.camera.get_width() / 2)
    kP = 1.0
    error = (contour_center - (rc.camera.get_width() / 2)) * scale
    angle = rc_utils.clamp(kP * error, -1, 1)

def turn(turnAngle: float) :
    global angle
    angle = turnAngle

def followWall():
    global speed, angle, lidarScan
    scan = rc.lidar.get_samples()
    left_angle, left_dist = rc_utils.get_lidar_closest_point(lidarScan, LEFT_WINDOW)
    right_angle, right_dist = rc_utils.get_lidar_closest_point(lidarScan, RIGHT_WINDOW)

    # rc.display.show_lidar(scan, 128, 1000, [(left_angle, left_dist), (right_angle, right_dist)])

    error = right_dist - left_dist  
    maxError = 12
    kP = 0.5

    angle = rc_utils.clamp(kP * error / maxError, -1, 1)
    speed = DRIVE_SPEED

    # speed = rc_utils.clamp(math.cos(0.5 * math.pi * angle) * DRIVE_SPEED  + MIN_SPEED, -1, 1) # smoothened version of -abs(angle) + 1
    # https://www.desmos.com/calculator/24qctllaj1
    
    print("Error: " + str(error))


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
