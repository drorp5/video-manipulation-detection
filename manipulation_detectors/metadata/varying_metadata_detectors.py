from .abstract_metadata_detector import *
from .varying_metadata import *

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

    @timeit
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
    
    def __init__(self, tolerance):
        self.estimated_timestamp_sec = 0
        self.estimated_rate_sec = 0
        self.proportional_error_coef = 1
        self.differential_error_coef = 0
        super().__init__(tolerance)

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
