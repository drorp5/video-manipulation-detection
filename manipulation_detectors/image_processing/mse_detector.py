from .abstract_image_processing_detector import *
from skimage import metrics


class MSEImageDetector(ImageProcessingDetector):
    "Detector based on mean squared error distance to check if identical"
    def __init__(self, min_th: float) -> None:
        self.min_th = min_th
        self.current_rgb_img = None
        self.prev_rgb_img = None

    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.IDENTICAL_DETECTED
    
    @timeit
    def validate(self) -> ManipulationDetectionResult:
        if self.prev_rgb_img is None:
             return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        score = metrics.mean_squared_error(self.current_rgb_img, self.prev_rgb_img)
        if score < self.min_th:
            return ManipulationDetectionResult(score, False, self.fake_status)
        return ManipulationDetectionResult(score, True, FakeDetectionStatus.REAL)
    
    @property
    def name(self) -> str:
        return "MSE"
