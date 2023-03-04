from __future__ import annotations
from .abstract_image_processing_detector import *

DEFAULT_HUE_BINS = 50
DEFAULT_SATURATION_BINS = 60
MAX_PIXEL_VAL = 255
MAX_SATURATION = 255
MAX_HUE = 179

@dataclass
class Histogram:
    hist: np.ndarray

    def normalize(self, axis=None) -> None:
        """ Normalize histogram. """
        self.hist = self.hist / np.sum(self.hist, axis=axis)

    def histograms_distance(self, other: Histogram) -> float:
        """ Measure distance between two histograms."""
        return cv2.compareHist(self.hist, other.hist, method=3)

def hue_saturation_histogram(rgb_img: np.ndarray, hue_bins: int = DEFAULT_HUE_BINS, saturation_bins: int = DEFAULT_SATURATION_BINS) -> Histogram:
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
    def __init__(self, min_th: float, hue_bins: int = DEFAULT_HUE_BINS, saturation_bins: int = DEFAULT_SATURATION_BINS):
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

    @timeit
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
     
