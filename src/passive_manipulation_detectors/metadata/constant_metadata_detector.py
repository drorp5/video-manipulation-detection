from .abstract_metadata_detector import *
from .constant_metadata import *

POSITIVE_RESULT = 0
NEGATIVE_RESULT = 1

class ConstantMetadataDetector(MetadataDetector):    
    @property
    def fake_status(self) -> FakeDetectionStatus:
        return FakeDetectionStatus.CONSTANT_METADATA_FAILURE
    
    def pre_process(self, frame: Frame):
        self.current_metadata = extract_constant_metadata(frame)

    @timeit    
    def validate(self) -> ManipulationDetectionResult:
        if self.prev_metadata is None:
            return ManipulationDetectionResult(NEGATIVE_RESULT, True, FakeDetectionStatus.FIRST)
        if self.current_metadata != self.prev_metadata:  # TODO: check if compared like this
            return ManipulationDetectionResult(NEGATIVE_RESULT, False, self.fake_status)
        return ManipulationDetectionResult(POSITIVE_RESULT, True, FakeDetectionStatus.REAL)

    @property
    def name(self) -> str:
        return 'ConstantMetadata'