import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import time
#import tflite_runtime.interpreter as tflite

import Queue

class ObjectDetector:
    def __init__(self):
        # print("tensorflow GPUs:",tf.config.list_physical_devices('GPU'))
        
        with tf.device('/GPU:0'):
            self.detect_fn = tf.saved_model.load("/simulator/startup_workspace/src/startup_package/src/models/saved_model")
            
            #Traffic Sign Recognition interpreter
            self.tsr_interpreter = tf.lite.Interpreter(model_path='/simulator/startup_workspace/src/startup_package/src/models/object_recognition_quant.tflite')
            self.tsr_interpreter.allocate_tensors()
            self.tsr_input_details = self.tsr_interpreter.get_input_details()
            self.tsr_output_details = self.tsr_interpreter.get_output_details()
    
    def getObjects(self,img):
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img_bright = self.increase_brightness(img, value=50)
        img_bright = img_rgb

        start = time.clock()  
        obj_list = self.objectDetection(img_bright)

        threshold=0.3
        for i, score in enumerate(obj_list['detection_scores']):
            if score >= threshold:
                w = img.shape[1]
                h = img.shape[0]
                ymin, xmin, ymax, xmax = obj_list['detection_boxes'][i]
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)

                # TF classes start from 1, ours start from 0
                obj_list['detection_classes'][i] = obj_list['detection_classes'][i] - 1.0
                roi = img_bright[ymin:ymax, xmin:xmax]
                #run the traffic sign recognition function,
                #which returns the new class ID
                obj_list['detection_classes'][i] = self.signRecognition(roi)


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

    def objectDetection_tflite(self, img_in):
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

    ############################## OBJECT DETECTION WITHOUT TFLITE ##############################
    def preprocess_image(self,img_in):
        HEIGHT = 320
        WIDTH = 320
        img = tf.convert_to_tensor(img_in)
        original_image = img
        img = tf.image.convert_image_dtype(img, tf.float32)
        resized_img = tf.image.resize(img, (HEIGHT, WIDTH))
        resized_img = resized_img[tf.newaxis, :]
        resized_img = tf.image.convert_image_dtype(resized_img, tf.uint8)
        return resized_img, original_image

    def clean_detections(self,detections):
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        # detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections

    def objectDetection(self,img_in):
        resized, original = self.preprocess_image(img_in)

        #start_time = time.monotonic()
        detections = self.detect_fn(resized)
        #end_time = time.monotonic()

        detections_clean = self.clean_detections(detections)

        return detections_clean
    


    ############################################################################################# 
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