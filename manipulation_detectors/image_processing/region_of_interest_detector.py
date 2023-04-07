from .abstract_image_processing_detector import *
from pathlib import Path
import json
from skimage import metrics

class RegionOfInterestDetector(ImageProcessingDetector):
    "Detector based on constant, repetitive region of interest such as the front shield."
    
    def __init__(self, min_th: float, roi_json: Path) -> None:
        self.min_th = min_th
        self.mask = binary_mask_from_json(roi_json)
        self.current_rgb_img = None
        self.prev_rgb_img = None

    @property
    def fake_status(self) -> FakeDetectionStatus:
        FakeDetectionStatus.ROI_MISMATCH

    def pre_process(self, rgb_img: np.ndarray) -> None:
        if rgb_img.shape != self.mask.shape:
            width = int(self.mask.shape[1])
            height = int(self.mask.shape[0])
            rgb_img = cv2.resize(rgb_img, (width, height))
        self.current_rgb_img = cv2.bitwise_and(rgb_img, rgb_img, mask=self.mask)
        
    @timeit
    def validate(self) -> ManipulationDetectionResult:
        if self.prev_rgb_img is None:
             return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        score = metrics.mean_squared_error(self.current_rgb_img, self.prev_rgb_img)
        if score > self.min_th:
            return ManipulationDetectionResult(score, False, self.fake_status)
        return ManipulationDetectionResult(score, True, FakeDetectionStatus.REAL)

    @property
    def name(self) -> str:
        return "RegionOfInterest"

def binary_mask_from_json(roi_json: Path) -> np.ndarray:
    """ Turn annotations json to binary mask.
        Credit to https://github.com/maftouni/binary_mask_from_json.git
        https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html
    """    
    with open(roi_json.as_posix(), "r") as read_file:
        data = json.load(read_file)
    region_of_interest_points = np.stack((data["roi_x"], data["roi_y"]), axis=1)
    mask = np.zeros((data["height"], data["width"]), dtype="uint8")
    mask = cv2.drawContours(mask, [region_of_interest_points], -1, 255, -1)
    return mask
