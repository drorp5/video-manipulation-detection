from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class FakeDetectionStatus(Enum):
    REAL = 'OK'
    FIRST = 'Not compared to anything'
    CONSTANT_METADATA_FAILURE = "The constant metadata does not match the previous frame"
    FRAME_ID_FAILURE = 'Frame ID Mismatch'
    TIMESTAMP_FAILURE = 'Timestamp Mismatch'
    TIMESTAMP_RATE_FAILURE = 'Timestamp Mismatch'
    IDENTICAL_DETECTED = 'Identical Image Detected'
    HISTOGRAM_MISMATCH = "Hue Saturation Histogram Mismatch"
    COMBINED = "Any of the other failure messages"
    

@dataclass
class ManipulationDetectionResult():
    score: float
    passed: bool
    message: FakeDetectionStatus


class ManipulationDetector(ABC):
    """Abstract class for manipulation detector"""
    @abstractmethod
    def validate(self) -> ManipulationDetectionResult:
        pass

    @property
    @abstractmethod
    def fake_status(self) -> FakeDetectionStatus:
        pass

    @abstractmethod
    def post_process(self) -> None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
