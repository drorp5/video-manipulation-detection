from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import pandas as pd
from vimba import Frame, PixelFormat
import numpy as np
import cv2
from config import CV2_CONVERSIONS, MAX_PIXEL_VAL, MAX_SATURATION, MAX_HUE
from skimage import metrics
from transmission_mock import *
from sign_detectors.stop_sign_detectors import HaarDetector

class FakeDetectionStatus(Enum):
    REAL = 'OK'
    FIRST = 'Not compared to anything'
    CONSTANT_METADATA_FAILURE = "The constant metadata does not match the previous frame"
    FRAME_ID_FAILURE = 'Frame ID Mismatch'
    TIMESTAMP_FAILURE = 'Timestamp Mismatch'
    TIMESTAMP_RATE_FAILURE = 'Timestamp Mismatch'
    IDENTICAL_DETECTED = 'Identical Image Detected'
    HISTOGRAM_MISMATCH = "Hue Saturation Histogram Mismatch"
    COMBINED = "Any of the other failure messages"
    

def nanoseconds_to_seconds(nano_sec: float) -> float:
    return nano_sec / 1e9

@dataclass
class ManipulationDetectionResult():
    score: float
    passed: bool
    message: FakeDetectionStatus
    

class ManipulationDetector(ABC):
    """Abstract class for manipulation detector"""
    @abstractmethod
    def validate(self) -> ManipulationDetectionResult:
        pass

    @property
    @abstractmethod
    def fake_status(self) -> FakeDetectionStatus:
        pass

    @abstractmethod
    def post_process(self) -> None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

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
    timestamp_seconds : float

def extract_varying_metadata(frame: Frame) -> FrameVaryingMetadata:
    frame_id = frame.get_id()
    timestamp_seconds = nanoseconds_to_seconds(frame.get_timestamp())
    return FrameVaryingMetadata(frame_id=frame_id, timestamp_seconds=timestamp_seconds)

class MetadataDetector(ManipulationDetector):
    def __init__(self):
        self.current_metadata = None
        self.prev_metadata = None
    
    @abstractmethod
    def pre_process(self, frame: Frame) -> None:
        pass

    def post_process(self) -> None:
        self.prev_metadata = self.current_metadata
        self.current_metadata = None

    def detect(self, frame: Frame) -> Tuple[bool, str]:
        self.pre_process(frame)
        detection_result = self.validate()
        self.post_process()
        is_real_img = detection_result.passed
        message = detection_result.message.value
        return is_real_img, message

class ConstantMetadataDetector(MetadataDetector):    
    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.CONSTANT_METADATA_FAILURE
    
    def pre_process(self, frame: Frame):
        self.current_metadata = extract_constant_metadata(frame)
        
    def validate(self) -> ManipulationDetectionResult:
        if self.prev_metadata is None:
            return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        if self.current_metadata != self.prev_metadata:  # TODO: check if compared like this
            return ManipulationDetectionResult(0, False, self.fake_status)
        return ManipulationDetectionResult(1, True, FakeDetectionStatus.REAL)

    @property
    def name(self) -> str:
        return 'ConstantMetadata'

class VaryingMetadataDetector(MetadataDetector):
    """Abstract class for detection based on incremental frame metadata"""
    def __init__(self, tolerance):
        self.tolerance = tolerance
        super().__init__()
    
    def pre_process(self, frame: Frame) -> None:
        self.current_metadata = extract_varying_metadata(frame)

    @abstractmethod
    def calc_score(self) -> float:
        pass

    def validate(self) -> ManipulationDetectionResult:
        if self.prev_metadata is None:
            return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        score = self.calc_score()
        if score >  self.tolerance:
            return ManipulationDetectionResult(score, False, self.fake_status)
        return ManipulationDetectionResult(score, True, FakeDetectionStatus.REAL)

class FrameIDDetector(VaryingMetadataDetector):
    """Detection using frame ID"""
    def __init__(self, tolerance: int = 1):
        super().__init__(tolerance)
    
    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.FRAME_ID_FAILURE
    
    def calc_score(self) -> float:
        current_frame_id = self.current_metadata.frame_id
        prev_frame_id = self.prev_metadata.frame_id
        return  abs(current_frame_id - prev_frame_id)   
    
    @property
    def name(self) -> str:
        return "FrameID"

class TimestampDetector(VaryingMetadataDetector):
    """Detection using frame relative timestamp"""
    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.TIMESTAMP_FAILURE

    def calc_score(self) -> float:
        current_timestamp = self.current_metadata.timestamp_seconds
        prev_timestamp = self.prev_metadata.timestamp_seconds
        return abs(current_timestamp - prev_timestamp)
    
    @property
    def name(self) -> str:
        return "Timestamp"
        
class TimestampRateDetector(VaryingMetadataDetector):
    """Detection using timestamp change rate"""
    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.TIMESTAMP_RATE_FAILURE

    def calc_score(self) -> float: 
        #TODO: implement
        pass

    @property
    def name(self) -> str:
        return "TimestampRate"
      
class ImageProcessingDetector(ManipulationDetector):
    "Abstract class for detection based on image processing techniques"

    @abstractmethod
    def pre_process(self, rgb_img: np.ndarray) -> None:
        pass
    
    def detect(self, rgb_img: np.ndarray) -> Tuple[bool, str]:
        self.pre_process(rgb_img)
        detection_result = self.validate()
        self.post_process()
        is_real_img = detection_result.passed
        message = detection_result.message.value
        return is_real_img, message

class MSEImageDetector(ImageProcessingDetector):
    "Detector based on mean squared error distance to check if identical"
    def __init__(self, min_th: float) -> None:
        self.min_th = min_th
        self.current_rgb_img = None
        self.prev_rgb_img = None

    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.IDENTICAL_DETECTED

    def pre_process(self, rgb_img: np.ndarray) -> None:
        self.current_rgb_img = rgb_img
    
    def validate(self) -> ManipulationDetectionResult:
        if self.prev_rgb_img is None:
             return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        score = metrics.mean_squared_error(self.current_rgb_img, self.prev_rgb_img)
        if score < self.min_th:
            return ManipulationDetectionResult(score, False, self.fake_status)
        return ManipulationDetectionResult(score, True, FakeDetectionStatus.REAL)

    def post_process(self) -> None:
        self.prev_rgb_img = self.current_rgb_img
        self.current_rgb_img = None
    
    @property
    def name(self) -> str:
        return "MSE"

@dataclass
class Histogram:
    hist: np.ndarray

    def normalize(self, axis=None) -> None:
        """ Normalize histogram. """
        self.hist = self.hist / np.sum(self.hist, axis=axis)

    def histograms_distance(self, other : Histogram) -> float:
        """ Measure distance between two histograms."""
        return cv2.compareHist(self.hist, other.hist, method=3)

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
        self.prev_hist = None
        
    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.HISTOGRAM_MISMATCH

    def pre_process(self, rgb_img: np.ndarray) -> None:
        self.current_hist = hue_saturation_histogram(rgb_img, hue_bins=self.hue_bins, saturation_bins=self.saturation_bins)
        self.current_hist.normalize()

    def validate(self) -> ManipulationDetectionResult:
        if self.prev_hist is None:
             return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        score = self.current_hist.histograms_distance(self.prev_hist)
        if score > self.min_th:
            return ManipulationDetectionResult(score, False, self.fake_status)
        return ManipulationDetectionResult(score, True, FakeDetectionStatus.REAL)

    def post_process(self) -> None:
        self.prev_hist = self.current_hist
        self.current_hist = None
    
    @property
    def name(self) -> str:
        return "Histogram"
     
def gvsp_frame_to_rgb(frame: Frame, cv2_transformation_code: int =  CV2_CONVERSIONS[PixelFormat.BayerRG8]):
    """Extract RGB image from gvsp frame object"""
    img = frame.as_opencv_image()
    rgb_img = cv2.cvtColor(img, cv2_transformation_code)
    return rgb_img

class CombinedDetector(ManipulationDetector):
    """Comibe metadata detectors and image procssing detectors to detect fake frames."""
    def __init__(self, metadata_detectors: List[MetadataDetector], image_processing_detectors: List[ImageProcessingDetector]):
        self.metadata_detectors = metadata_detectors
        self.image_processing_detectors = image_processing_detectors

    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.COMBINED

    @property
    def name(self) -> str:
        return "Combined"


    def pre_process(self, frame: Frame) -> None:
        for detector in self.metadata_detectors:
            detector.pre_process(frame)
        rgb_img = gvsp_frame_to_rgb(frame)
        # plot_rgb(rgb_img)
        for detector in self.image_processing_detectors:
            detector.pre_process(rgb_img)

    def validate(self) -> ManipulationDetectionResult:
        for detector in self.metadata_detectors + self.image_processing_detectors:
            detector_status = detector.validate()
            if not detector_status.passed:
                return detector_status
        return ManipulationDetectionResult(None, True, FakeDetectionStatus.REAL)

    def post_process(self) -> None:
        for detector in self.metadata_detectors + self.image_processing_detectors:
            detector.post_process()

    def detect(self, frame: Frame) -> Tuple[bool, str]:
        self.pre_process(frame)
        detection_result = self.validate()
        self.post_process()
        is_real_img = detection_result.passed
        message = detection_result.message.value
        return is_real_img, message
    
    def validate_experiments(self) -> Tuple[bool, Dict[str, float, List[FakeDetectionStatus]]]:
        results = {}
        passed = True
        status = []
        for detector in self.metadata_detectors + self.image_processing_detectors:
            detector_status = detector.validate()
            results[detector.name] = detector_status.score
            if not detector_status.passed:
                passed = False
                status.append(detector_status.message)
        if len(status) == 0:
            status.append(FakeDetectionStatus.REAL)
        return passed, results, status

    def detect_experiments(self, frame: Frame) -> Tuple[bool, Dict, List[FakeDetectionStatus]]:
        self.pre_process(frame)
        passed, detection_results, status = self.validate_experiments()
        self.post_process()
        return passed, detection_results, status


def plot_rgb(rgb_img: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(rgb_img)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\live_stream_defaults_start.pcapng"
    gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\faking_matlab_rec_3.pcapng"
    
    
    gvsp_transmission = MockGvspTransmission(gvsp_pcap_path=gvsp_path)
    
    constant_metadata_detector = ConstantMetadataDetector()
    frame_id_detector = FrameIDDetector()
    timestamp_detector = TimestampDetector(0.1 * 3)

    histogram_detector = HueSaturationHistogramDetector(0.4)
    mse_detector = MSEImageDetector(0.01)
    
    combined_detector = CombinedDetector([constant_metadata_detector, frame_id_detector, timestamp_detector], [mse_detector, histogram_detector])
    
    stop_sign_detector = HaarDetector()

    scores = []
    stop_sign_detection = []
    for frame in gvsp_transmission.frames:
        if frame is not None:
            passed, detection_results, status = combined_detector.detect_experiments(frame)
            scores.append(detection_results)
            ic(passed)
            if not(passed):
                ic(status)
            # ic(combined_detector.detect_experiments(frame))

            detections = stop_sign_detector.detect(gvsp_frame_to_rgb(frame))
            stop_sign_detection.append(len(detections) > 0)
            
    scores_df = pd.DataFrame(scores)
    fig, axs = plt.subplots(6,1)
    axs[0].plot(stop_sign_detection)
    axs[0].set_ylabel('stop sign')
    ax_id = 1
    for name, res in scores_df.iteritems():
        ax = axs[ax_id]
        res.plot(ax=axs[ax_id])
        ax.grid(True)
        ax.set_ylabel(name)
        ax_id +=1
    plt.show()
    print(passed)