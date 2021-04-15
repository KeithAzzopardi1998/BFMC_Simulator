import cv2
import numpy as np
import tensorflow as tf
import time
#import tflite_runtime.interpreter as tflite


class ObjectDetector:
    def __init__(self):
        print("tensorflow GPUs:",tf.config.list_physical_devices('GPU'))
        
        with tf.device('/GPU:0'):
            #Object Detection interpreter
            self.od_interpreter = tf.lite.Interpreter(model_path='/simulator/startup_workspace/src/startup_package/src/models/object_detector_quant_4.tflite')
            self.od_interpreter.allocate_tensors()
            self.od_input_details = self.od_interpreter.get_input_details()
            self.threshold = 0.2

            #Traffic Sign Recognition interpreter
            self.tsr_interpreter = tf.lite.Interpreter(model_path='/simulator/startup_workspace/src/startup_package/src/models/object_recognition_quant.tflite')
            self.tsr_interpreter.allocate_tensors()
            self.tsr_input_details = self.tsr_interpreter.get_input_details()
            self.tsr_output_details = self.tsr_interpreter.get_output_details()
            self.threshold = 0.2
    
    def getObjects(self,img):
        #image_brightened = self.increase_brightness(img, value=30)

        start = time.clock()  
        obj_list = self.objectDetection(img)
        end = time.clock()
        print("object detection took", end-start)

        #looping through the list of objects, and updating
        #the class ID of any traffic signs
        for o in obj_list:
            #the main OD model uses class 7 for traffic signs
            if o['class_id']== 7.0:
                #grab the part of the image containing the sign
                w = img.shape[1]
                h = img.shape[0]
                ymin, xmin, ymax, xmax = o['bounding_box']
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)
                roi = img[ymin:ymax, xmin:xmax]
                #run the traffic sign recognition function,
                #which returns the new class ID
                o['class_id'] = self.signRecognition(roi)
        
        return obj_list

    def set_input_od(self, image):
        with tf.device('/GPU:0'):
            """Sets the input tensor."""
            tensor_index = self.od_input_details[0]['index']
            input_tensor = self.od_interpreter.tensor(tensor_index)()[0]
            input_tensor[:, :] = image
    
    def set_input_tsr(self, image):
        with tf.device('/GPU:0'):
            """Sets the input tensor."""
            tensor_index = self.tsr_input_details[0]['index']
            input_tensor = self.tsr_interpreter.tensor(tensor_index)()[0]
            input_tensor[:, :] = image

    def get_output_tensor(self, index):
        with tf.device('/GPU:0'):
            """Returns the output tensor at the given index."""
            output_details = self.od_interpreter.get_output_details()[index]
            tensor = np.squeeze(self.od_interpreter.get_tensor(output_details['index']))
            return tensor

    def make_result(self, box, class_id, scores):
        result = {
                    'bounding_box': box,
                    'class_id': class_id,
                    'score': scores
        }
        return result

    def objectDetection(self, img_in):
        with tf.device('/GPU:0'):
            input_shape = self.od_input_details[0]['shape']
            _, height, width, _ = input_shape

            resized_image = cv2.resize(img_in, (width, height), interpolation=cv2.INTER_LINEAR)

            resized_image = resized_image[np.newaxis, :]

            start = time.clock()
            self.set_input_od(resized_image)
            end = time.clock()
            #print("setting input took", end-start)

            start = time.clock()
            self.od_interpreter.invoke()
            end = time.clock()
            #print("invoke took", end-start)

            boxes = np.clip(self.get_output_tensor(0), 0, 1)
            classes = self.get_output_tensor(1)
            scores = self.get_output_tensor(2)
            count = int(self.get_output_tensor(3))

            start = time.clock()
            results = [self.make_result(boxes[i], classes[i], scores[i]) for i in range(count) if scores[i] >= self.threshold]
            end = time.clock()
            #print("reading output took", end-start)

            #print(results)

            return results

    def signRecognition(self,img_in):
        with tf.device('/GPU:0'):
            input_shape = self.tsr_input_details[0]['shape']
            _, height, width, _ = input_shape

            resized_image = cv2.resize(img_in, (width, height), interpolation=cv2.INTER_LINEAR)

            resized_image = resized_image[np.newaxis, :]

            self.set_input_tsr(resized_image)

            self.tsr_interpreter.invoke()

            output_proba = self.tsr_interpreter.get_tensor(self.tsr_output_details[0]['index'])[0]
            tsr_class = np.argmax(output_proba)

            return float("7.%d"%tsr_class)

    def increase_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

        return img