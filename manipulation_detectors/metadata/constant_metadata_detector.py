from .abstract_metadata_detector import *
from .constant_metadata import *

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