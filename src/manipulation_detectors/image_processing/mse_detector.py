from .abstract_image_processing_detector import *
from skimage import metrics


class MSEImageDetector(ImageProcessingDetector):
    "Detector based on mean squared error distance to check if identical"
    def __init__(self, min_th: float, max_th:float = np.inf) -> None:
        #TODO: add max_th
        self.current_rgb_img = None
        self.prev_rgb_img = None
        super().__init__(min_th = min_th, max_th=max_th)

    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.IDENTICAL_DETECTED
    
    @timeit
    def validate(self) -> ManipulationDetectionResult:
        if self.prev_rgb_img is None:
             return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        score = metrics.mean_squared_error(self.current_rgb_img, self.prev_rgb_img)
        if  self.min_th <= score <= self.max_th:
            return ManipulationDetectionResult(score, True, FakeDetectionStatus.REAL)
        return ManipulationDetectionResult(score, False, self.fake_status)
    
    @property
    def name(self) -> str:
        return "MSE"
