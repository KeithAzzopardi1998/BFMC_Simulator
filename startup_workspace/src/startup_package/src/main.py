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
from autonomous_controller import AutonomousController
from path_planning import PathPlanner

import rospy
import cv2
import numpy as np
import random
from time import sleep

import threading

import Queue
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

def getImage_pp(img):
	return img.copy()

# ===================================== LANE DETECTION VISUALIZATION ===============================
def get_vertices_for_img(img):
	img_shape = img.shape
	height = img_shape[0]
	width = img_shape[1]

	region_top_left     = (0.00*width, 0.30*height)
	region_top_right    = (1.00*width, 0.30*height)
	region_bottom_left  = (0.00*width, 1.00*height)
	region_bottom_right = (1.00*width, 1.00*height)

	vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
	return vert

def draw_lines(img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
	# Copy the passed image
	img_copy = np.copy(img) if make_copy else img

	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

	return img_copy

def trace_lane_line_with_coefficients(img, line_coefficients, top_y, make_copy=True):
	A = line_coefficients[0]
	b = line_coefficients[1]
	if A==0.0 and b==0.0:
		img_copy = np.copy(img) if make_copy else img
		return img_copy

	height, width,_ = img.shape
	bottom_y = height - 1
	# y = Ax + b, therefore x = (y - b) / A
	bottom_x = (bottom_y - b) / A
	# clipping the x values
	bottom_x = min(bottom_x, 2*width)
	bottom_x = max(bottom_x, -1*width)

	top_x = (top_y - b) / A
	# clipping the x values
	top_x = min(top_x, 2*width)
	top_x = max(top_x, -1*width)

	new_lines = [[[int(bottom_x), int(bottom_y), int(top_x), int(top_y)]]]
	return draw_lines(img, new_lines, make_copy=make_copy)

def drawIntersectionLine(img, y_intercept, make_copy=True):
	_, width,_ = img.shape
	line = [[[0, int(y_intercept), width, int(y_intercept)]]]
	return draw_lines(img, line,color=[0, 255, 0], make_copy=make_copy)

def getImage_ld(image_in, lane_info, intersection_y):
	vert = get_vertices_for_img(image_in)
	left_coefficients = lane_info[0]
	right_coefficients = lane_info[1]
	region_top_left = vert[0][1]

	lane_img_left = trace_lane_line_with_coefficients(image_in, left_coefficients, region_top_left[1], make_copy=True)

	if intersection_y == -1:
		lane_img_final = trace_lane_line_with_coefficients(lane_img_left, right_coefficients, region_top_left[1], make_copy=False)
	else:
		lane_img_both = trace_lane_line_with_coefficients(lane_img_left, right_coefficients, region_top_left[1], make_copy=True)
		lane_img_final = drawIntersectionLine(lane_img_both,intersection_y, make_copy=False)

	# image1 * alpha + image2 * beta + lambda
	# image1 and image2 must be the same shape.
	img_with_lane_weight =  cv2.addWeighted(image_in, 0.7, lane_img_final, 0.3, 0.0)

	return img_with_lane_weight

# ===================================== OBJECT DETECTION VISUALIZATION ===============================
LABEL_DICT = {0: 'bike',
                10: 'bus',
                20: 'car',
                30: 'motor',
                40: 'person',
                50: 'rider',
                60: 'traffic_light',
                70: 'ts_priority',
                71: 'ts_stop',
                72: 'ts_no_entry',
                73: 'ts_one_way',
                74: 'ts_crossing',
                75: 'ts_fw_entry',
                76: 'ts_fw_exit',
                77: 'ts_parking',
                78: 'ts_roundabout',
                80: 'train',
                90: 'truck'
            }

COLORS = np.random.randint(0, 255, size=(len(LABEL_DICT), 3), dtype="uint8")

def getImage_od(img,obj_info):
	# based on https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py
	#print("getImage_od: type is",type(obj_info))
	if not obj_info:
		#print('list was empty')
		return img
	else:
		threshold = 0.3
		for i, score in enumerate(obj_info['detection_scores']):
			if score >= threshold:
				w = img.shape[1]
				h = img.shape[0]
				ymin, xmin, ymax, xmax = obj_info['detection_boxes'][i]
				xmin = int(xmin * w)
				xmax = int(xmax * w)
				ymin = int(ymin * h)
				ymax = int(ymax * h)

				idx = int(obj_info['detection_classes'][i]*10)
				#print("idx has type",type(idx))
				# Skip the background
				#if idx >= len(LABEL_DICT):
				#	continue

				cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
				y = ymin - 15 if ymin - 15 > 15 else ymin + 15
				cv2.putText(img,"{}: {:.2f}%".format(LABEL_DICT[idx], score * 100),
							(xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

	return img

# ===================================== MAIN CODE ===============================

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

plan = PathPlanner()
print("Path Planner loaded")

bno = BNO055()
print("BNO055 loaded")

ld = LaneDetector()
print("Lane Detector created")

od = ObjectDetector()
print("Object Detector created")

con = AutonomousController()
print("AutonomousController created")

#creating a thread for updating the current path progress
def updateCurrentLocation():
	while True:
		try:
			#update the current location
			gps_data = gps.getGpsData()
			if gps_data:
				coord_complex = gps_data['coor'][0]
				x = coord_complex.real
				y = coord_complex.imag
				#print("gps position: (%.3f,%.3f)"%(x,y))
				plan.updatePathProgress(x,y)
				sleep(0.5)
		except Exception as e:
			print("Error updating path progress:",e)

gps_thread = threading.Thread(target=updateCurrentLocation, name="gps_thread", args=())
gps_thread.daemon = True
gps_thread.start()

steering = 0.0
speed = 0.0

objects = []
while 1:
	#raw image
	img_in = cam.getImage()
	height = img_in.shape[0]
	width = img_in.shape[1]
	
	#preprocessed image
	#(to handle lighting etc)
	img_pp = getImage_pp(img_in)

	#detect lanes
	lanes, intersection,lane_preprocessed_img = ld.getLanes(img_pp.copy())
	#print("lanes:",lanes)
	#print("intersection y:",intersection_y)
	img_ld = getImage_ld(img_pp.copy(),lanes, intersection[1])
	#img_ld = img_pp.copy()
	#lane_preprocessed_img = img_pp.copy()

	objects = od.getObjects(img_pp.copy())
	img_od = getImage_od(img_pp.copy(),objects)

	#visualize the detections
	img_in_resized = cv2.resize(img_in,(int(width/2),int(height/2)))

	#TODO revert this
	#img_pp_resized = cv2.resize(img_pp,(int(width/2),int(height/2)))
	img_pp_resized = cv2.resize(lane_preprocessed_img,(int(width/2),int(height/2)))

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
	con.chooseRoutine(lanes, intersection, objects, (img_in.shape[0],img_in.shape[1]))


print("Car stopped. \n END")
car.stop(0.0)
