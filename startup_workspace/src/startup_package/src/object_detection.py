import cv2
import numpy
import tensorflow as tf

class ObjectDetector:
    def __init__(self):
        print("tensorflow GPUs:",tf.config.list_physical_devices('GPU'))
    
    def getObjects(self,img):
        return []