from abc import ABC, abstractmethod
from typing import List
import cv2 
import numpy as np
from pathlib import Path

MAX_PIXEL_VALUE = 255
NUM_CHANNELS = 3

MODELS_DIR = Path('sign_detectors/')

class StopSignDetector(ABC):
    @abstractmethod
    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError
    
def draw_bounding_boxes(img, bounding_boxes):
    for (x,y,w,h) in bounding_boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (MAX_PIXEL_VALUE, MAX_PIXEL_VALUE, 0), NUM_CHANNELS)
    return img

def list_detectors() -> List[str]:
        return list(get_detectors_dict().keys())

def get_detectors_dict():
    return {"Haar": HaarDetector, 
            "Yolo": YoloDetector,
            "MobileNet": MobileNetDetector}

def get_detector(detector_name: str) -> StopSignDetector:
    if detector_name is None:
        return None
    detectors_dict = get_detectors_dict()
    if detector_name not in detectors_dict:
        raise ValueError
    return detectors_dict[detector_name]()


class HaarDetector(StopSignDetector):
    def __init__(self, grayscale=False, blur=False):
        self.config_path = MODELS_DIR / 'stop_sign_classifier_2.xml'
        self.config_path = self.config_path.as_posix()
        self.detector = cv2.CascadeClassifier(self.config_path)
        self.grayscale = grayscale
        self.blur = blur

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        if self.blur:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        stop_signs = self.detector.detectMultiScale(img, scaleFactor=1.05, minNeighbors=15, minSize=(30, 30))
        return stop_signs
    
    @property
    def name(self):
        return "Haar"


class YoloDetector(StopSignDetector):
    def __init__(self, confidence_th=0.5, nms_th=0.4 ):
        coco_names_path = MODELS_DIR / 'coco.names'
        with open(coco_names_path.as_posix(), 'r') as f:
            self.classes = f.read().strip().split('\n')
        self.target_class = [9, 11]
        self.config_path = MODELS_DIR / 'yolov4-tiny.cfg'
        self.weights_path = MODELS_DIR / 'yolov4-tiny.weights'
        self.detector = cv2.dnn.readNetFromDarknet(self.config_path.as_posix(), self.weights_path.as_posix())
        ln = self.detector.getLayerNames()
        self.ln = [ln[i - 1] for i in self.detector.getUnconnectedOutLayers()]
        self.inference_shape = (416,416)
        self.confidence_th = confidence_th
        self.nms_th = nms_th

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        blob = cv2.dnn.blobFromImage(img, 1/MAX_PIXEL_VALUE, self.inference_shape, swapRB=True, crop=False)
        self.detector.setInput(blob)
        outputs = self.detector.forward(self.ln)

        boxes = []
        confidences = []
        h, w = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                if classID in self.target_class:
                    confidence = scores[classID]
                    if confidence > self.confidence_th:
                        box = detection[:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        box = [x, y, int(width), int(height)]
                        boxes.append(box)
                        confidences.append(float(confidence))
        # boxes = np.array(boxes)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_th, self.nms_th)
        # if len(indices) > 0:
        # indices = indices.squeeze()
        boxes = [boxes[i] for i in indices]
        return boxes
    
    @property
    def name(self):
        return "Yolo"


class MobileNetDetector(StopSignDetector):
    def __init__(self, confidence_th=0.5):
        coco_names_path = MODELS_DIR / 'coco.names'
        with open(coco_names_path.as_posix(), 'r') as f:
            self.classes = f.read().strip().split('\n')
        self.target_class = [10,11,13]
        self.config_path = MODELS_DIR / 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.weights_path = MODELS_DIR / 'ssd_mobilenet_v3_largefrozen_inference_graph.pb'
        self.detector = cv2.dnn_DetectionModel(self.weights_path.as_posix(), self.config_path.as_posix())
        self.inference_shape = (480,480)
        self.detector.setInputSize(self.inference_shape[0], self.inference_shape[1]) #greater this value better the results tune it for best output
        self.detector.setInputScale(1.0/(MAX_PIXEL_VALUE/2))
        self.detector.setInputMean((MAX_PIXEL_VALUE/2, MAX_PIXEL_VALUE/2, MAX_PIXEL_VALUE/2))
        self.detector.setInputSwapRB(True)
        self.confidence_th = confidence_th
    
    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        boxes = []
        detections_class_index, detections_confidence, detections_bbox = self.detector.detect(img, confThreshold=self.confidence_th)
        if len(detections_class_index) == 0:
            return boxes
        for class_ind, confidence, detection_box in zip(detections_class_index.flatten(), detections_confidence.flatten(), detections_bbox):
            if class_ind in self.target_class and confidence >= self.confidence_th:
                boxes.append(detection_box)
        return boxes
        
    @property
    def name(self):
        return "MobileNet"