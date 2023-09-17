from __future__ import annotations
from dataclasses import dataclass, field
import json
import re
from typing import List, Tuple
import numpy as np
import pandas as pd
from enum import Enum

@dataclass
class IntensityExposureFrame:
    """Simplified frame object with id, exposure time and intensity"""
    id: int
    exposure: float
    intensity: float
    

@dataclass
class ExposureChange:
    """Stores data of frame in which there was an exposure time change"""
    cur_frame: IntensityExposureFrame
    prev_frame: IntensityExposureFrame
    
    @property
    def id(self):
        return self.cur_frame.id
    
    @property
    def exposure_difference(self):
        return self.cur_frame.exposure - self.prev_frame.exposure
    
    @property
    def prev_exposure(self):
        return self.prev_frame.exposure
    

class ExposureValidationStatus(Enum):
    EVALUATION = 0
    SUCCESS = 1
    FAIL = 2
    INCOMPLETE = 3
    

@dataclass
class ExposureChangeDetectionResult:
    change: ExposureChange
    status: ExposureValidationStatus
    matching: IntensityExposureFrame = None


class ExposureChangeValidator(): #TODO: consider split to two Validator and Validation(Validator, ExposureFrame)
    def __init__(self, exposure_change: ExposureChange, max_offset: int, min_offset: int=5) -> None:
        self.exposure_change = exposure_change
        self.max_offset = max_offset
        self.min_offset = min_offset
        self.checked_offsets = set()
        self.max_missing_intensity_frames = 0
        self._are_all_options_exhausted = False

        self.ratio_function_parameters = np.array([6.34566360e+04, 1.69825909e-02])

    def ratio_function(self, x):
        # self.ratio_function_parameters = np.array([6.34566360e+04, 1.69825909e-02])
        # return self.ratio_function_parameters[0] / x**2 + self.ratio_function_parameters[1]
        
        # a = 98254.92004913
        # return np.where(x<2000, a / x**2, (2000)**(0.5-2) *a / x**0.5)
        a,b = -9.21015457e-05,  1.74208437e-01
        return np.where(x<1500, a*x+b,  (a*1500+b) * (1500/x)**0.5)
    
    def estimate_intensity_diff_to_exposure_diff_ratio(self, exposure: float) -> Tuple[float, float]:
        ratio = self.ratio_function(exposure)
        err = 0.015
        return ratio, err

    def are_valid_intensity_frames(self, cur_frame_id: int, prev_frame_id) -> bool:
        return 0 < cur_frame_id - prev_frame_id <= self.max_missing_intensity_frames + 1

    def is_valid_offset(self, offset: int) -> bool:
        return self.min_offset <= offset <= self.max_offset
    
    def is_valid_intensity_diff(self, intensity_diff):
        return not np.isnan(intensity_diff)
             
    def is_intensity_diff_matches_estimation(self, intensity_diff) -> bool:
        abs_diff = abs(self.expected_intensity_diff - intensity_diff) 
        return abs_diff <= self.expected_intensity_diff_err

    @property
    def ratio(self):
        return self.estimate_intensity_diff_to_exposure_diff_ratio(self.exposure_change.prev_frame.exposure)[0]
    
    @property
    def ratio_err(self):
        return self.estimate_intensity_diff_to_exposure_diff_ratio(self.exposure_change.prev_frame.exposure)[1]
    
    @property
    def expected_intensity_diff(self):
        return self.ratio * self.exposure_change.exposure_difference
    
    @property
    def expected_intensity_diff_err(self):
        return abs(self.ratio_err * self.exposure_change.exposure_difference)
    

    def validate(self, cur_frame: IntensityExposureFrame, prev_frame: IntensityExposureFrame) -> bool:
        if cur_frame is None or prev_frame is None:
            return False
        if not self.are_valid_intensity_frames(cur_frame.id, prev_frame.id):
            return False
        offset = cur_frame.id - self.exposure_change.id 
        self.update_are_all_options_exhausted(offset)
        if not self.is_valid_offset(offset):
            return False
        if offset in self.checked_offsets:
            return False
        self.checked_offsets.add(offset)
        intensity_diff = cur_frame.intensity - prev_frame.intensity
        return self.is_intensity_diff_matches_estimation(intensity_diff)
        
    @property
    def are_all_options_exhausted(self) -> bool:
        return self._are_all_options_exhausted
    
    @are_all_options_exhausted.setter
    def are_all_options_exhausted(self, value):
        self._are_all_options_exhausted = value
    
    def update_are_all_options_exhausted(self, offset:int):
        if not self.are_all_options_exhausted:
            self.are_all_options_exhausted = offset >= self.max_offset
    
    @property
    def are_all_options_tested(self):
        return len(self.checked_offsets) == self.max_offset - self.min_offset + 1

class ExposureTimeChangeDetector: 
    def __init__(self, max_offset: int = 20):
        self.max_offset = max_offset
        self.cur_frame = None
        self.last_intensity_frame = None
        self.last_exposure_frame = None
        self.changes_validations_buffer = []
        self.max_missing_exposure_frames = 1

    def is_valid_exposure_diff(self) -> bool:
        if self.last_exposure_frame is None:
            return False
        return self.cur_frame.id - self.last_exposure_frame.id <= self.max_missing_exposure_frames + 1

    def calc_exposure_diff(self) -> np.float:
        if self.is_valid_exposure_diff():
           return self.cur_frame.exposure - self.last_exposure_frame.exposure
        return np.nan

    def add_exposure_change(self) -> None:
        if self.is_valid_exposure_diff():
            exposure_difference = self.calc_exposure_diff()
            if not np.isnan(exposure_difference) and exposure_difference != 0:
                exposure_change = ExposureChange(cur_frame=self.cur_frame,prev_frame=self.last_exposure_frame)
                exposure_change_validator = ExposureChangeValidator(exposure_change, self.max_offset)
                self.changes_validations_buffer.append((exposure_change, exposure_change_validator))

    def update_last_exposure_frame(self) -> None:
        if not np.isnan(self.cur_frame.exposure):
            self.last_exposure_frame = self.cur_frame
    
    def update_last_intensity_frame(self) -> None:
        if not np.isnan(self.cur_frame.intensity):
            self.last_intensity_frame = self.cur_frame

    def validate_change(self) -> ExposureChangeDetectionResult:
        if len(self.changes_validations_buffer) > 0:
            exposure_change, exposure_change_validation = self.changes_validations_buffer[0]
            is_matching = exposure_change_validation.validate(self.cur_frame, self.last_intensity_frame) 
            if is_matching:
                self.changes_validations_buffer.pop(0)
                return ExposureChangeDetectionResult(change=exposure_change, 
                                                      status=ExposureValidationStatus.SUCCESS,
                                                      matching=self.cur_frame)
            if exposure_change_validation.are_all_options_exhausted:
                self.changes_validations_buffer.pop(0)
                if exposure_change_validation.are_all_options_tested:
                     return ExposureChangeDetectionResult(change=exposure_change, 
                                                      status=ExposureValidationStatus.FAIL)
                return ExposureChangeDetectionResult(change=exposure_change, 
                                                      status=ExposureValidationStatus.INCOMPLETE)
            return ExposureChangeDetectionResult(change=exposure_change, 
                                                status=ExposureValidationStatus.EVALUATION)
            
    def feed_frame(self, frame_id: int, exposure: float, intensity: float) -> ExposureChangeDetectionResult:
        self.cur_frame = IntensityExposureFrame(frame_id, exposure, intensity)
        self.add_exposure_change()
        detection_result = self.validate_change()
        self.update_last_exposure_frame()
        self.update_last_intensity_frame()
        return detection_result

if __name__ == "__main__":
    import adaptive_parameters.utils as utils
    from pathlib import Path
    
    adaptive_parameters_path = Path('INPUT\\10_8_23\\adaptive_parameters_2023_08_10_15_35_12.json')
    frames_dir = Path('OUTPUT\\10_8_23\\recording_2023_08_10_15_35_12_images')
    
    frames_exposure_id, frames_exposure_time = utils.read_exposure_data(adaptive_parameters_path)
    exposure_df = pd.DataFrame({'exposure': frames_exposure_time}, index=frames_exposure_id)
    frames_intensity_id, frames_intensity = utils.read_intensity_data(frames_dir)
    intensity_df = pd.DataFrame({'intensity': frames_intensity}, index=frames_intensity_id)
    exposure_intensity_df = pd.merge(exposure_df, intensity_df, how="outer", left_index=True, right_index=True)
    
    detector = ExposureTimeChangeDetector()
    changes_detected = {}
    for frame_id, frame_data in exposure_intensity_df[3:].iterrows():
        detection_result = detector.feed_frame(frame_id, frame_data['exposure'], frame_data['intensity'])
        if detection_result is not None and detection_result.status != ExposureValidationStatus.EVALUATION:
            changes_detected[detection_result.change.id] = detection_result
    
    print(f'# detected = {len([k for k,v in changes_detected.items() if v.status==ExposureValidationStatus.SUCCESS])}')
    print(f'# not detected = {len([k for k,v in changes_detected.items() if v.status==ExposureValidationStatus.FAIL])}')
    print(f'# not completed = {len([k for k,v in changes_detected.items() if v.status==ExposureValidationStatus.INCOMPLETE])}')

    print('finished')