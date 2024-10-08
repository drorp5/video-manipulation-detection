from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from .abstract_detector import *
from .metadata import *
from .image_processing import *
from gige.gvsp_frame import gvsp_frame_to_rgb

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

    @timeit
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
    
    def validate_experiments(self) -> Dict[str, ManipulationDetectionResult]:
        results = {}
        for detector in self.metadata_detectors + self.image_processing_detectors:
            detector_result = detector.validate()
            results[detector.name] = detector_result
        return results

    def detect_experiments(self, frame: Frame) -> Dict[str, ManipulationDetectionResult]:
        self.pre_process(frame)
        results = self.validate_experiments()
        self.post_process()
        return results