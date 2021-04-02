#!/usr/bin/python

"""
These are the libraries you need to import in your project in order to
be able to communicate with the Gazebo simulator
"""
from bfmclib.gps_s import Gps
from bfmclib.bno055_s import BNO055
from bfmclib.camera_s import CameraHandler
from bfmclib.controller_p import Controller
from bfmclib.trafficlight_s import TLColor, TLLabel, TrafficLight

from lane_detection import LaneDetector
from object_detection import ObjectDetector

import rospy
import cv2
import numpy as np
import random
from time import sleep

def getImage_pp(img):
	return img.copy()

def getImage_ld(img,lane_info):
	return img.copy()

def getImage_od(img,obj_info):
	return img.copy()

# This line should be the first line in your program
rospy.init_node('main_node', anonymous=True)

cam = CameraHandler()
print("Camera loaded")

car = Controller()
print("Controller loaded")

sem = TrafficLight()
print("Traffic lights listener")

gps = Gps()
print("Gps loaded")

bno = BNO055()
print("BNO055 loaded")

ld = LaneDetector()
print("Lane Detector created")

od = ObjectDetector()
print("Object Detector created")

car.drive(-0.1, 0.0)
sleep(3)

steering = 0.0
speed = 0.0

while 1:
	#raw image
	img_in = cam.getImage()
	height = img_in.shape[0]
	width = img_in.shape[1]
	
	#preprocessed image
	#(to handle lighting etc)
	img_pp = getImage_pp(img_in)

	#detect lanes
	lanes = ld.getLanes(img_pp.copy())
	print("lanes:",lanes)
	img_ld = getImage_ld(img_pp,lanes)
	
	#detect objects
	objects = od.getObjects(img_pp.copy())
	img_od = getImage_od(img_pp,objects)

	#visualize the detections
	img_in_resized = cv2.resize(img_in,(int(width/2),int(height/2)))
	img_pp_resized = cv2.resize(img_pp,(int(width/2),int(height/2)))
	img_ld_resized = cv2.resize(img_ld,(int(width/2),int(height/2)))
	img_od_resized = cv2.resize(img_od,(int(width/2),int(height/2)))

	img_out = np.vstack((
		np.hstack((img_in_resized,img_pp_resized)),
		np.hstack((img_ld_resized,img_od_resized))
	))

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img_out,'Raw',		        (0,int(height*0.49)),                     font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img_out,'Preprocessed',     (int(width*0.5),int(height*0.49)),   font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img_out,'Lanes',   			(0,int(height*0.99)),                     font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(img_out,'Objects',     		(int(width*0.5),int(height*0.99)),   font,1,(255,255,255),2,cv2.LINE_AA)

	cv2.imshow("Frame preview", img_out)
	key = cv2.waitKey(1)
	if key == ord('q'):
		cv2.destroyAllWindows()
		break

	#use the detected lanes and objects to make decisions
	steering = random.uniform(-25,25)
	speed = 0.2
	print("Sending move with speed %d, steering %d"%(speed,steering))
	car.drive(speed, steering)
	sleep(0.01)

print("Car stopped. \n END")
car.stop(0.0)
