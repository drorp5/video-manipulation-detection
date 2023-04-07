""" Based on open-cv docs on optical flow
    https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
"""

from pathlib import Path
from .abstract_image_processing_detector import *


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))
    
class OpticalFlowDetector(ImageProcessingDetector):
    "Detector based on optical flow mismatch"
    def __init__(self, min_th: float) -> None:
        self.min_th = min_th
        self.current_rgb_img: np.ndarray = None
        self.current_gray_img: np.ndarray = None
        self.current_features: np.ndarray = None
        self.prev_gray_img: np.ndarray = None
        self.prev_features: np.ndarray = None

        self.good_old = None
        self.good_new = None
        self.img_shape = None
        
    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.OPTICAL_FLOW_MISMATCH
    
    def pre_process(self, rgb_img: np.ndarray) -> None:
        self.current_rgb_img = rgb_img
        self.current_gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        self.current_features = cv2.goodFeaturesToTrack(self.current_gray_img, mask=None, **feature_params)
        self.img_shape = rgb_img.shape
    
    @timeit
    def validate(self) -> ManipulationDetectionResult:
        if self.prev_features is None:
            return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        
        #TODO: consider using self.current_features as initial estimation instead of None
        next_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray_img, self.current_gray_img, self.prev_features, None, **lk_params)
        if next_points is not None:
            good_new = next_points[status==1]
            self.current_features = good_new.reshape(-1,1,2)
            good_old = self.prev_features[status==1]
            err = err[status==1]
            score = np.mean(abs(err))
            self.good_new = good_new
            self.good_old = good_old

            if score > self.min_th:
                return ManipulationDetectionResult(score, False, self.fake_status)
        else:
            score = np.nan
        return ManipulationDetectionResult(score, True, FakeDetectionStatus.REAL)

    def post_process(self) -> None:
        self.prev_gray_img = self.current_gray_img
        self.prev_features = self.current_features
        self.current_rgb_img = None
        self.current_gray_img = None
        self.current_features = None

    @property
    def name(self) -> str:
        return "OpticalFlow"
    
    def draw_optical_flow_tracks(self) -> np.ndarray:
        mask = np.zeros(shape=self.img_shape, dtype=np.uint8)
        for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            mask = cv2.circle(mask, (int(a), int(b)), 5, color[i].tolist(), -1)
        return mask