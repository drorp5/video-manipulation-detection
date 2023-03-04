from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import numpy as np
import time
class FakeDetectionStatus(Enum):
    REAL = 'OK'
    FIRST = 'Not compared to anything'
    CONSTANT_METADATA_FAILURE = "The constant metadata does not match the previous frame"
    FRAME_ID_FAILURE = 'Frame ID Mismatch'
    TIMESTAMP_FAILURE = 'Timestamp Mismatch'
    TIMESTAMP_RATE_FAILURE = 'Timestamp Mismatch'
    IDENTICAL_DETECTED = 'Identical Image Detected'
    HISTOGRAM_MISMATCH = "Hue Saturation Histogram Mismatch"
    OPTICAL_FLOW_MISMATCH = "Optical flow of interest point mismatch"
    COMBINED = "Any of the other failure messages"
    

@dataclass
class ManipulationDetectionResult():
    score: float
    passed: bool
    message: FakeDetectionStatus
    _process_time_sec: float = np.nan

    @property
    def process_time_sec(self) -> float:
        return self._process_time_sec

    @process_time_sec.setter
    def process_time_sec(self, process_time_sec: float):
        self._process_time_sec = process_time_sec


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        result.process_time_sec = total_time
        return result
    return timeit_wrapper

class ManipulationDetector(ABC):
    """Abstract class for manipulation detector"""
    @abstractmethod
    @timeit
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
