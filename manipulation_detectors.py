from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from vimba import Frame, PixelFormat
import numpy as np
import cv2
from config import CV2_CONVERSIONS, MAX_PIXEL_VAL, MAX_SATURATION, MAX_HUE
from skimage import metrics

class FakeDetectionStatus(Enum):
    REAL = 'OK'
    WIDTH_FAILURE = 'Width Mismatch'
    HEIGHT_FAILURE = 'Height Mismatch'
    PIXEL_FORMAT_FAILURE = 'Pixel Format Mismatch'
    FRAME_ID_FAILURE = 'Frame ID Mismatch'
    TIMESTAMP_FAILURE = 'Timestamp Mismatch'
    IDENTICAL_DETECTED = 'Identical Image Detected'
    HISTOGRAM_MISMATCH = "Hue Saturation Histogram Mismatch"
    

@dataclass
class FrameConstantMetadata:
    width : int
    height : int
    pixel_format : PixelFormat

def extract_constant_metadata(frame: Frame) -> FrameConstantMetadata:
    width = frame.get_width()
    height = frame.get_height()
    pixel_format = frame.get_pixel_format()
    return FrameConstantMetadata(width, height, pixel_format)

@dataclass
class FrameVaryingMetadata:
    frame_id : int
    timestamp : float # TODO: check and add units

def extract_varying_metadata(frame: Frame) -> FrameVaryingMetadata:
    frame_id = frame.get_id()
    timestamp = frame.get_timestamp()
    return FrameVaryingMetadata(frame_id=frame_id, timestamp=timestamp)

class MetadataDetector(ABC):
    def __init__(self):
        self.current_metadata = None
        self.prev_metadata = None
    
    @abstractmethod
    def pre_process(self, frame: Frame) -> None:
        pass

    @abstractmethod
    def validate(self) -> FakeDetectionStatus:
        pass
    
    def post_process(self) -> None:
        self.prev_metadata = self.current_metadata
        self.current_metadata = None


class ConstantMetadataDetector(MetadataDetector):    
    def pre_process(self, frame: Frame):
        self.current_metadata = extract_constant_metadata(frame)
        
    def validate(self) -> FakeDetectionStatus:
        if self.prev_metadata is None:
            return FakeDetectionStatus.REAL
        if self.current_metadata.width != self.prev_metadata.width:
            return FakeDetectionStatus.WIDTH_FAILURE
        if self.current_metadata.height != self.prev_metadata.height:
            return FakeDetectionStatus.HEIGHT_FAILURE
        if self.current_metadata.pixel_format != self.prev_metadata.pixel_format:  # TODO: check if compared like this
            return FakeDetectionStatus.PIXEL_FORMAT_FAILURE
        return FakeDetectionStatus.REAL

        
class VaryingMetadataDetector(MetadataDetector):
    def __init__(self, timestamp_tolerance : float, frame_id_tolerance: int = 1): # TODO: check units of timestamp and add to name
        self.frame_id_tolerance = frame_id_tolerance
        self.timestamp_tolerance = timestamp_tolerance
        super().__init__()
    
    def pre_process(self, frame: Frame) -> None:
        self.current_metadata = extract_varying_metadata(frame)

    def validate(self) ->FakeDetectionStatus:
        if self.prev_metadata is None:
            return FakeDetectionStatus.REAL
        if not self.check_frame_id():
            return FakeDetectionStatus.FRAME_ID_FAILURE
        if not self.check_timestamp():
            return FakeDetectionStatus.TIMESTAMP_FAILURE
        return FakeDetectionStatus.REAL
        
    def check_frame_id(self) -> bool:
        current_frame_id = self.current_metadata.frame_id
        prev_frame_id = self.prev_metadata.frame_id
        return abs(current_frame_id - prev_frame_id) < self.frame_id_tolerance
    
    def check_timestamp(self) -> bool:
        current_timestamp = self.current_metadata.timestamp
        prev_timestamp = self.prev_metadata.timestamp
        return abs(current_timestamp - prev_timestamp) < self.timestamp_tolerance

    def check_timestamp_rate(self, frame: Frame) -> bool:
        #TODO: implement      
        return True


@dataclass
class Histogram:
    hist: np.ndarray

    def normalize(self, axis=None) -> None:
        """ Normalize histogram. """
        self.hist = self.hist / np.sum(self.hist, axis=axis)

    def histograms_similarity(self, other : Histogram) -> float:
        """ Measure similarity between two histograms."""
        return cv2.compareHist(self.hist, other.hist, method=3)
        
class ImageProcessingDetector(ABC):
    """Abstract class for detection based on classic image processing techniques"""
    @abstractmethod
    def pre_process(self, rgb_img: np.ndarray) -> None:
        pass    

    @abstractmethod
    def validate(self) -> FakeDetectionStatus:
        pass

    @abstractmethod
    def post_process(self) -> None:
        pass

class MSEImageDetector(ImageProcessingDetector):
    "Detector based on mean squared error distance to check if identical"
    def __init__(self, min_th: float) -> None:
        self.min_th = min_th
        self.current_rgb_img = None
        self.prev_rgb_img = None

    def pre_process(self, rgb_img: np.ndarray) -> None:
        self.current_rgb_img = rgb_img
    
    def validate(self) -> FakeDetectionStatus:
        if self.prev_rgb_img is None:
            return FakeDetectionStatus.REAL
        mean_sqared_error = metrics.mean_squared_error(self.current_rgb_img, self.prev_rgb_img)
        if mean_sqared_error < self.min_th:
            return FakeDetectionStatus.IDENTICAL_DETECTED
        return FakeDetectionStatus.REAL

    def post_process(self) -> None:
        self.prev_rgb_img = self.current_rgb_img
        self.current_rgb_img = None

def hue_saturation_histogram(rgb_img: np.ndarray, hue_bins: int = 50, saturation_bins: int = 60) -> Histogram:
    """Calculate Hue-Saturation Histogram."""
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    hist_size = [hue_bins, saturation_bins]
    h_ranges = [0, MAX_HUE + 1]
    s_ranges = [0, MAX_SATURATION + 1]
    ranges = h_ranges + s_ranges 
    channels = [0, 1]
    hist = cv2.calcHist([hsv_img], channels, None, hist_size, ranges, accumulate=False)
    return Histogram(hist)

class HueSaturationHistogramDetector(ImageProcessingDetector):
    """Detector based on histogram of the hue-saturation channels"""
    def __init__(self, min_th: float, hue_bins: int = 50, saturation_bins: int = 60 ):
        self.min_th = min_th
        self.hue_bins = hue_bins
        self.saturation_bins = saturation_bins
        
    def pre_process(self, rgb_img: np.ndarray) -> None:
        self.current_hist = hue_saturation_histogram(rgb_img, hue_bins=self.hue_bins, saturation_bins=self.saturation_bins)
        self.current_hist.normalize()

    def validate(self) -> FakeDetectionStatus:
        hist_similarity = self.current_hist.histograms_similarity(self.prev_hist)
        if hist_similarity < self.min_th:
            return FakeDetectionStatus.HISTOGRAM_MISMATCH
        return FakeDetectionStatus.REAL

    def post_process(self) -> None:
        self.prev_hist = self.current_hist
        self.current_hist = None
     
def gvsp_frame_to_rgb(frame: Frame, cv2_transformation_code: CV2_CONVERSIONS = None): #TODO: change conversion code
    """Extract RGB image from gvsp frame object"""
    img = frame.as_opencv_image()
    rgb_img = cv2.cvtColor(img, cv2_transformation_code)
    return rgb_img

class CombinedDetector():
    """Comibe metadata detectors and image procssing detectors to detect fake frames."""
    def __init__(self, metadata_detectors: List[MetadataDetector], image_proessing_detectors: List[ImageProcessingDetector]):
        self.metadata_detectors = metadata_detectors
        self.image_proessing_detectors = image_proessing_detectors

    def pre_process(self, frame: Frame) -> None:
        for detector in self.metadata_detectors:
            detector.pre_process(frame)
        rgb_img = gvsp_frame_to_rgb(frame)
        for detector in self.image_proessing_detectors:
            detector.pre_process(rgb_img)

    def validate(self) -> FakeDetectionStatus:
        for detector in self.metadata_detectors + self.image_proessing_detectors:
            detector_status = detector.validate()
            if not detector_status == FakeDetectionStatus.REAL:
                return detector_status
        return FakeDetectionStatus.REAL
        
    def post_process(self) -> None:
        for detector in self.metadata_detectors + self.image_proessing_detectors:
            detector.post_process()

    def detect(self, frame: Frame) -> Tuple[bool, str]:
        self.pre_process(frame)
        fake_detection_status = self.validate()
        self.post_process()
        is_real_img = fake_detection_status == FakeDetectionStatus.REAL
        message = fake_detection_status.value
        return is_real_img, message
    

if __name__ == "__main__":
    path1 = r"INPUT/stop_sign_road.jpg"
    path2 = r"INPUT/stop_sign_road_2.jpg"
    path3 = r"INPUT/stop_sign_road_3.jpg"

    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.imread(path3)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

    enriched_image_1 = EnrichedImage(img1)    
    enriched_image_2 = EnrichedImage(img2)
    enriched_image_3 = EnrichedImage(img3)

    print(enriched_image_1.compare_histograms(enriched_image_2))
    print(enriched_image_1.compare_histograms(enriched_image_3))
    print(enriched_image_2.compare_histograms(enriched_image_3))
