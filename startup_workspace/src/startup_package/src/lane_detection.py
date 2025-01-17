import cv2
import numpy as np
import preprocessing as pp

class LaneDetector:
    def __init__(self):
        self.img_shape = (480, 640)
        self.height = self.img_shape[0]
        self.width = self.img_shape[1]

        #use this code for a hexagonal ROI
        #   region_top_left = (0.15*self.width, 0.3*self.height)
        #   region_top_right = (0.85*self.width, 0.3*self.height)
        #   region_bottom_left_A = (0.00*self.width, 1.00*self.height)
        #   region_bottom_left_B = (0.00*self.width, 0.8*self.height)
        #   region_bottom_right_A = (1.00*self.width, 1.00*self.height)
        #   region_bottom_right_B = (1.00*self.width, 0.8*self.height)
        #   self.mask_vertices = np.array([[region_bottom_left_A,
        #                                    region_bottom_left_B,
        #                                    region_top_left,
        #                                    region_top_right,
        #                                    region_bottom_right_B,
        #                                    region_bottom_right_A]], dtype=np.int32)

        region_top_left = (0.2*self.width, 0.6*self.height)
        region_top_right = (0.8*self.width, 0.6*self.height)
        region_bottom_left = (0.00*self.width, 1.00*self.height)
        region_bottom_right = (1.00*self.width, 1.00*self.height)

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
            #return np.array([[0.0,0.0], [0.0,0.0]], img_in
            left_lane_lines = []
            right_lane_lines = []
            horizontal_lines = []
            preprocessed_img = img_in

        #this function returns the y-intercept of the intersection
        #if one is found, else it returns -1
        intersection_info = self.check_for_intersection(horizontal_lines)

        try:
            left_lane_slope, left_intercept = pp.getLanesFormula(left_lane_lines)  
            #print("left slope:",left_lane_slope)      
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [left_lane_slope, left_intercept])
        except Exception as e:
            #print("Using saved coefficients for left coefficients", e)
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [0.0, 0.0])
            
        try: 
            right_lane_slope, right_intercept = pp.getLanesFormula(right_lane_lines)
            #print("right slope:",right_lane_slope)
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [right_lane_slope, right_intercept])
        except Exception as e:
            #print("Using saved coefficients for right coefficients", e)
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [0.0, 0.0])

        #return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]),intersection_y, preprocessed_img
        return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]),intersection_info, preprocessed_img

    def check_for_intersection(self,lines):
        # if there are no horizontal lines, there is definitely no intersection
        if not lines:
            return [-1,-1]

        # to check if there is an intersection, we first calculate
        # the length of each line
        # each line is of the format [x1, y1, x2, y2]
        # we use the difference of x values (instead of pythagoras)
        # as it is faster, and we know that the lines are of a low gradient
        line_lengths = np.array([ abs(l[0] - l[2]) for l in lines])

        # this is the "consensus function" which determines whether
        # there is an intersection or not
        cond1 = (np.mean(line_lengths) >= (self.width/3))
        cond2 = (len(lines)>10)
        cond3 = (len([ l for l in line_lengths if l >=(self.width*0.75)]) 
                        >= len(line_lengths)*0.5)
        detected = cond1 or cond2 or cond3

        if detected:
            print("detected intersection with condition(s) c1: %s, c2: %s, c3: %s"
                    % (cond1,cond2,cond3))
            slope, intercept = pp.getLanesFormula(lines)
            return [slope, intercept]
        else:
            return [-1,-1]
