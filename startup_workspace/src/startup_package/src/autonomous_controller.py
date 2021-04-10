import cv2
import numpy as np
from bfmclib.controller_p import Controller
import math
import time

class AutonomousController:
    def __init__(self):
        self.car = Controller()
        self.angle_weights = np.array([0.8, 0.15, 0.05])
        self.angles_to_store = 3
        self.last_n_angles = np.zeros(self.angles_to_store)
        self.index = 0
        
        #TODO get these from the XML path planner
        self.actions = ["right","straight","left","left"]
        self.action_counter = 0


    def chooseRoutine(self, lane_info, intersection_info, obj_info,frame_size):
        try:
            ld_left, ld_right = lane_info
            int_grad, int_y = intersection_info

            if int_y >= 320:
                self.routine_intersection(int_grad, int_y)

            self.routine_cruise(ld_left, ld_right,frame_size)
            
        except Exception as e:
            print("AutonomousController failed:\n",e,"\n\n")
    
    def routine_intersection(self,intersection_grad,intersection_y):
        print("routine_intersection: STOPPING AT INTERSECTION AT Y=%d"%intersection_y)
        print("the gradient is ",intersection_grad)
        self.car.stop(0.0)
        time.sleep(3)

        #the angle at which we are approaching the intersection (in degrees)
        theta = math.atan(intersection_grad)
        alpha = 90 + math.atan(intersection_grad)
        print("the angle of approach is ",alpha)

        # TODO: how to determine these?
        next_action = self.getNextAction()
        
        if next_action=="right":
            print("routine_intersection: making right turn")
            self.car.drive(0.15, 0.0 + (theta*3))
            time.sleep(3)
            self.car.drive(0.15, 20)
            time.sleep(4)
        elif next_action=="left":
            print("routine_intersection: making left turn")
            self.car.drive(0.15, 0.0 + (theta*3))
            time.sleep(3)
            self.car.drive(0.15, 0.0)
            time.sleep(3)
            self.car.drive(0.15, -20)
            time.sleep(5)
        elif next_action=="straight":
            print("routine_intersection: going straight")
            self.car.drive(0.15, 0.0 + (theta*3))
            time.sleep(3)
            self.car.drive(0.15, 0.0)
            time.sleep(3)
            self.car.drive(0.15, 0.0)
            time.sleep(3)
        else:
            pass
        
        time.sleep(0.1)

    def getNextAction(self):
        a = self.actions[self.action_counter]
        self.action_counter +=1
        return a

    def routine_cruise(self,lane_left,lane_right,frame_size):
        steering_angle = self.calculate_steering_angle(lane_left,lane_right,frame_size)
        
        self.last_n_angles[self.index % self.angles_to_store] = steering_angle

        weighted_angle = 0.0

        for i in range(self.angles_to_store):
            weighted_angle += self.last_n_angles[(self.index + i + 1) % self.angles_to_store] * self.angle_weights[i]

        #print('weighted angle', weighted_angle)

        self.index += 1
        if self.index % self.angles_to_store == 0 and self.index >= 20:
            self.index = 0

        self.car.drive(0.15, weighted_angle)
        time.sleep(0.1)
    
    def calculate_steering_angle(self,lane_left,lane_right,frame_size):
        #convert from lane lines to lane points
        left_lane_pts = self.points_from_lane_coeffs(lane_left,frame_size)
        right_lane_pts = self.points_from_lane_coeffs(lane_right,frame_size)

        height, width = frame_size
        x_offset = 0.0

        left_x1, left_y1, left_x2, left_y2 = left_lane_pts
        right_x1, right_y1, right_x2, right_y2 = right_lane_pts

        left_found = False if (left_x1==0 and left_y1==0 and left_x2==0 and left_y2==0) else True
        #if left_found: print("found left lane")
        right_found = False if (right_x1==0 and right_y1==0 and right_x2==0 and right_y2==0) else True
        #if right_found: print("found right lane")

        if left_found and right_found: #both lanes
            cam_mid_offset_percent = 0.02
            mid = int(width/2 * (1 + cam_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid
        elif left_found and not right_found: #left lane only
            x_offset = left_x2 - left_x1
        elif not left_found and right_found: #right lane ony
            x_offset = right_x2 - right_x1
        else: #no lanes detected
            x_offset = 0
        
        #HACK: this was /2 before
        y_offset = float(height/1.8)

        steering_angle = math.atan(x_offset / y_offset) #in radians
        steering_angle = steering_angle * 180.0 / math.pi
        steering_angle = np.clip(steering_angle, -15.0, 15.0)
        return steering_angle
    
    def points_from_lane_coeffs(self,line_coefficients,frame_size):
        A = line_coefficients[0]
        b = line_coefficients[1]

        if A==0.00 and b==0.00:
            return [0,0,0,0]

        height, width = frame_size

        bottom_y = height - 1
        #this should be where the LaneDetector mask ends
        top_y = 0.6 * height
        # y = Ax + b, therefore x = (y - b) / A
        bottom_x = (bottom_y - b) / A
        # clipping the x values
        bottom_x = min(bottom_x, 2*width)
        bottom_x = max(bottom_x, -1*width)

        top_x = (top_y - b) / A
        # clipping the x values
        top_x = min(top_x, 2*width)
        top_x = max(top_x, -1*width)

        return [int(bottom_x), int(bottom_y), int(top_x), int(top_y)]

