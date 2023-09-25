from .abstract_metadata_detector import *
from .varying_metadata import *
from datetime import datetime

class VaryingMetadataDetector(MetadataDetector):
    """Abstract class for detection based on incremental frame metadata"""
    def pre_process(self, frame: Frame) -> None:
        self.current_metadata = extract_varying_metadata(frame)

    @abstractmethod
    def calc_score(self) -> float:
        pass

    @timeit
    def validate(self) -> ManipulationDetectionResult:
        if self.prev_metadata is None:
            return ManipulationDetectionResult(0, True, FakeDetectionStatus.FIRST)
        score = self.calc_score()
        if self.min_th <= score <= self.max_th:
            return ManipulationDetectionResult(score, True, FakeDetectionStatus.REAL)
        return ManipulationDetectionResult(score, False, self.fake_status)

class FrameIDDetector(VaryingMetadataDetector):
    """Detection using frame ID"""
    def __init__(self, max_th: int = 1):
        super().__init__(max_th=max_th)
    
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
    
    def __init__(self, max_th):
        self.estimated_timestamp_sec = 0
        self.estimated_rate_sec = 0
        self.proportional_error_coef = 1
        self.differential_error_coef = 0
        super().__init__(max_th=max_th)

    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.TIMESTAMP_FAILURE

    def pre_process(self, frame: Frame) -> None:
        self.current_metadata = extract_varying_metadata(frame)
        self.estimated_timestamp_sec += self.estimated_rate_sec
        
    def calc_score(self) -> float:
        return abs(self.calc_error())
    
    def calc_error(self) -> float:
        current_timestamp = self.current_metadata.timestamp_seconds
        # prev_timestamp = self.prev_metadata.timestamp_seconds
        return current_timestamp - self.estimated_timestamp_sec
    
    def post_process(self) -> None:
        err = self.calc_error()
        self.estimated_timestamp_sec += self.proportional_error_coef * err
        return super().post_process()
        
    @property
    def name(self) -> str:
        return "Timestamp"
        
class TimestampRateDetector(VaryingMetadataDetector):
    """Detection using timestamp change rate vs reality """
    def __init__(self, min_th, max_th):
        self.prev_internal_timestamp = None
        self.current_internal_timestamp = None
        super().__init__(min_th=min_th, max_th=max_th)

    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.TIMESTAMP_RATE_FAILURE
    
    def pre_process(self, frame: Frame) -> None:
        self.current_metadata = extract_varying_metadata(frame)
        self.current_internal_timestamp = datetime.now()

    def calc_score(self) -> float: 
        external_timestamp_diff_seconds = self.current_metadata.timestamp_seconds - self.prev_metadata.timestamp_seconds
        internal_timestamp_diff = self.current_internal_timestamp - self.prev_internal_timestamp
        internal_timestamp_diff_seconds = internal_timestamp_diff.total_seconds()
        error = internal_timestamp_diff_seconds - external_timestamp_diff_seconds
        return abs(error)

    @property
    def name(self) -> str:
        return "TimestampRate"

    def post_process(self) -> None:
        self.prev_internal_timestamp = self.current_internal_timestamp
        super().post_process()