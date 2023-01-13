from typing import List
import cv2 
import numpy as np

class StopSignDetector:
    def __init__(self, config_path, grayscale=False, blur=False):
        self.stop_sign_cascade = cv2.CascadeClassifier(config_path)
        self.grayscale = grayscale
        self.blur = blur

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        if self.blur:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        stop_signs = self.stop_sign_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=15, minSize=(30, 30))
        return stop_signs

def draw_bounding_boxes(img, bounding_boxes):
    for (x,y,w,h) in bounding_boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)
    return img

