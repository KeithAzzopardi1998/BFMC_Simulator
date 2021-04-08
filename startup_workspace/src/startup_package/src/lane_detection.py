import cv2
import numpy as np
import preprocessing as pp

class LaneDetector:
    def __init__(self):
        self.img_shape = (480, 640)
        height = self.img_shape[0]
        width = self.img_shape[1]

        #use this code for a hexagonal ROI
        #   region_top_left = (0.15*width, 0.3*height)
        #   region_top_right = (0.85*width, 0.3*height)
        #   region_bottom_left_A = (0.00*width, 1.00*height)
        #   region_bottom_left_B = (0.00*width, 0.8*height)
        #   region_bottom_right_A = (1.00*width, 1.00*height)
        #   region_bottom_right_B = (1.00*width, 0.8*height)
        #   self.mask_vertices = np.array([[region_bottom_left_A,
        #                                    region_bottom_left_B,
        #                                    region_top_left,
        #                                    region_top_right,
        #                                    region_bottom_right_B,
        #                                    region_bottom_right_A]], dtype=np.int32)

        region_top_left = (0.2*width, 0.6*height)
        region_top_right = (0.8*width, 0.6*height)
        region_bottom_left = (0.00*width, 1.00*height)
        region_bottom_right = (1.00*width, 1.00*height)
        self.mask_vertices = np.array([[region_bottom_left,
                                         region_top_left,
                                         region_top_right,
                                         region_bottom_right]], dtype=np.int32)
    
    def getLanes(self, img_in):
        try:
            # Setting Hough Transform Parameters
            rho = 1 # 1 degree
            theta = (np.pi/180) * 1
            threshold = 15
            min_line_length = 20
            max_line_gap = 10

            left_lane_coefficients  = pp.create_coefficients_list()
            right_lane_coefficients = pp.create_coefficients_list()

            previous_left_lane_coefficients = None
            previous_right_lane_coefficients = None

            intersection_y = -1

            # Begin lane detection pipiline
            img = img_in.copy()
            img = cv2.convertScaleAbs(img,alpha=2.0,beta=30)
            #print("LANE LINES LOG - COPIED IMAGE", img)
            combined_hsl_img = pp.filter_img_hsl(img)
            #print("LANE LINES LOG - COMBINED IMAGE HSL", combined_hsl_img)
            grayscale_img = pp.grayscale(combined_hsl_img)
            #print("LANE LINES LOG - COMBINED IMAGE GRAYSCALE", grayscale_img)
            gaussian_smoothed_img = pp.gaussian_blur(grayscale_img, kernel_size=5)
            canny_img = cv2.Canny(gaussian_smoothed_img, 50, 150)
            segmented_img = pp.getROI(canny_img,self.mask_vertices)
            #print("LANE LINES LOG - SEGMENTED IMAGE SUM", np.sum(segmented_img))
            hough_lines = pp.hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)
            #print("LANE LINES LOG - HOUGH LINES", hough_lines)

            preprocessed_img = cv2.cvtColor(segmented_img,cv2.COLOR_GRAY2BGR)
            left_lane_lines, right_lane_lines, horizontal_lines = pp.separate_lines(hough_lines, img)

        except Exception as e:
            print("lane preprocessing failed")
            #return np.array([[0.0,0.0], [0.0,0.0]], img_in
            left_lane_lines = []
            right_lane_lines = []
            horizontal_lines = []
            preprocessed_img = img_in

        #TODO : check this threshold (it was determined using a very short test)
        if len(horizontal_lines) >= 10 :
            intersection_y = self.check_for_intersection(horizontal_lines)

        try:
            left_lane_slope, left_intercept = pp.getLanesFormula(left_lane_lines)        
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [left_lane_slope, left_intercept])
        except Exception as e:
            #print("Using saved coefficients for left coefficients", e)
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [0.0, 0.0])
            
        try: 
            right_lane_slope, right_intercept = pp.getLanesFormula(right_lane_lines)
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [right_lane_slope, right_intercept])
        except Exception as e:
            #print("Using saved coefficients for right coefficients", e)
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [0.0, 0.0])

        #return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]),intersection_y, preprocessed_img
        return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]),intersection_y, preprocessed_img

    def check_for_intersection(self,lines):
        #print("########### checking for intersection ###########")
        #for l in lines:
        #    print(l)
        slope, intercept = pp.getLanesFormula(lines)
        #print(slope)
        #print(intercept)
        #print("#################################################")
        return intercept
