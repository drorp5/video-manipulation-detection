from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import cv2
import numpy as np
from pathlib import Path

from utils.detection_utils import sliding_window, DetectedObject, non_maximal_supression

MAX_PIXEL_VALUE = 255
NUM_CHANNELS = 3

MODELS_DIR = Path("sign_detectors/")

INPUT_SIZE = (484, 304)  # width, height


class StopSignDetector(ABC):
    def __init__(self, confidence_th=0.5, nms_th=0.4):
        self.confidence_th = confidence_th
        self.nms_th = nms_th

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def _detect(self, img: np.ndarray) -> List[DetectedObject]:
        raise NotImplementedError

    def detect(self, img: np.ndarray) -> List[DetectedObject]:
        # sliding window detection
        window_size = INPUT_SIZE
        step_size = None  # no overlap

        # Process each window
        all_detections = []
        for window_x, window_y, window in sliding_window(img, window_size, step_size):
            window_detections = self._detect(window)
            for detection in window_detections:
                all_detections.append(detection.offset_by(window_x, window_y))
        supressed_detections = non_maximal_supression(
            detections=all_detections,
            confidence_th=self.confidence_th,
            nms_th=self.nms_th,
        )
        return supressed_detections


def draw_detections(
    img: np.ndarray, detections: List[DetectedObject], with_confidence: bool = False
) -> np.ndarray:
    out_img = img.copy()
    for detection in detections:
        cv2.rectangle(
            out_img,
            detection.get_upper_left_corner(),
            detection.get_lower_right_corner(),
            (0, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE),
            NUM_CHANNELS,
        )
        if with_confidence:
            confidence_text = f"{detection.confidence:.2f}"
            text_position = (
                max(detection.get_lower_right_corner()[0] - detection.width + 5, 0),
                min(
                    detection.get_upper_left_corner()[1] + detection.height - 5,
                    img.shape[0],
                ),
            )
            cv2.putText(
                out_img,
                confidence_text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (0, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE),  # Font color (yellow in BGR)
                1,  # Thickness
                cv2.LINE_AA,
            )

    return out_img


def list_detectors() -> List[str]:
    return list(get_detectors_dict().keys())


def get_detectors_dict():
    return {"Haar": HaarDetector, "Yolo": YoloDetector, "MobileNet": MobileNetDetector}


def get_detector(detector_name: str) -> StopSignDetector:
    if detector_name is None:
        return None
    detectors_dict = get_detectors_dict()
    if detector_name not in detectors_dict:
        raise ValueError
    return detectors_dict[detector_name]()


class HaarDetector(StopSignDetector):
    def __init__(self, confidence_th=0, nms_th=0, grayscale=False, blur=False):
        super().__init__(confidence_th, nms_th)
        self.config_path = MODELS_DIR / "stop_sign_classifier_2.xml"
        self.config_path = self.config_path.as_posix()
        self.detector = cv2.CascadeClassifier(self.config_path)
        self.grayscale = grayscale
        self.blur = blur
        self.input_size = INPUT_SIZE

    def _detect(self, img: np.ndarray) -> List[DetectedObject]:
        if self.blur:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stop_signs = self.detector.detectMultiScale(
            img, scaleFactor=1.05, minNeighbors=15, minSize=(30, 30)
        )
        return [DetectedObject(bounding_box) for bounding_box in stop_signs]

    @property
    def name(self):
        return "Haar"


class YoloDetector(StopSignDetector):
    def __init__(self, confidence_th=0.5, nms_th=0.4):
        super().__init__(confidence_th, nms_th)
        coco_names_path = MODELS_DIR / "coco.names"
        with open(coco_names_path.as_posix(), "r") as f:
            self.classes = f.read().strip().split("\n")
        self.target_class = [9, 11]
        self.config_path = MODELS_DIR / "yolov4-tiny.cfg"
        self.weights_path = MODELS_DIR / "yolov4-tiny.weights"
        self.detector = cv2.dnn.readNetFromDarknet(
            self.config_path.as_posix(), self.weights_path.as_posix()
        )
        ln = self.detector.getLayerNames()
        self.ln = [ln[i - 1] for i in self.detector.getUnconnectedOutLayers()]
        self.inference_shape = INPUT_SIZE

    def _detect(self, img: np.ndarray) -> List[DetectedObject]:
        blob = cv2.dnn.blobFromImage(
            img, 1 / MAX_PIXEL_VALUE, self.inference_shape, swapRB=True, crop=False
        )
        self.detector.setInput(blob)
        outputs = self.detector.forward(self.ln)

        h, w = img.shape[:2]
        detections = []
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
                        detections.append(
                            (
                                DetectedObject(
                                    bounding_box=[x, y, int(width), int(height)],
                                    confidence=float(confidence),
                                )
                            )
                        )

        return detections

    @property
    def name(self):
        return "Yolo"


class MobileNetDetector(StopSignDetector):
    def __init__(self, confidence_th=0.5, nms_th=0.4):
        super().__init__(confidence_th, nms_th)
        coco_names_path = MODELS_DIR / "coco.names"
        with open(coco_names_path.as_posix(), "r") as f:
            self.classes = f.read().strip().split("\n")
        self.target_class = [13]  # 10 is traffic light, I dont know what is 11
        self.config_path = MODELS_DIR / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        self.weights_path = (
            MODELS_DIR / "ssd_mobilenet_v3_largefrozen_inference_graph.pb"
        )
        self.detector = cv2.dnn_DetectionModel(
            self.weights_path.as_posix(), self.config_path.as_posix()
        )
        self.inference_shape = INPUT_SIZE
        self.detector.setInputSize(
            self.inference_shape[0], self.inference_shape[1]
        )  # greater this value better the results tune it for best output
        self.detector.setInputScale(1.0 / (MAX_PIXEL_VALUE / 2))
        self.detector.setInputMean(
            (MAX_PIXEL_VALUE / 2, MAX_PIXEL_VALUE / 2, MAX_PIXEL_VALUE / 2)
        )
        self.detector.setInputSwapRB(True)

    def _detect(self, img: np.ndarray) -> List[DetectedObject]:
        detections = []
        detections_class_index, detections_confidence, detections_bbox = (
            self.detector.detect(img, confThreshold=self.confidence_th)
        )
        if len(detections_class_index) == 0:
            return detections
        for class_ind, confidence, detection_box in zip(
            detections_class_index.flatten(),
            detections_confidence.flatten(),
            detections_bbox,
        ):
            if class_ind in self.target_class and confidence >= self.confidence_th:
                detections.append(DetectedObject(detection_box, confidence))
        return detections

    @property
    def name(self):
        return "MobileNet"
