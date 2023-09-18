from __future__ import annotations
from dataclasses import dataclass, field
import json
import re
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from enum import Enum
from numpy.typing import ArrayLike

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
    EVALUATION = 0  # detection still possible
    SUCCESS = 1     # detection succeeded 
    FAIL = 2        # detection failed
    INCOMPLETE = 3  # detection failed - not all options observed
    

@dataclass
class ExposureChangeDetectionResult:
    change: ExposureChange
    status: ExposureValidationStatus
    matching: Optional[IntensityExposureFrame] = None

@dataclass
class IntensityDiffToExposureDiffRatio:
    value: ArrayLike
    error: float = 0.015

    @staticmethod
    def init_of_exposure(exposure: ArrayLike, err: Optional[float] = None) -> IntensityDiffToExposureDiffRatio:
        a,b = -9.21015457e-05,  1.74208437e-01

        threshold_exposure = 1500
        lower_ratio_function = lambda  x: a*x+b
        upper_ratio_function = lambda x:  lower_ratio_function(threshold_exposure) * (threshold_exposure/x)**0.5

        ratio =  np.where(exposure<threshold_exposure, lower_ratio_function(exposure),  upper_ratio_function(exposure))

        if err is not None:
            return IntensityDiffToExposureDiffRatio(ratio, err)
        return IntensityDiffToExposureDiffRatio(ratio)    


class ExposureChangeValidator(): #TODO: consider split to two Validator and Validation(Validator, ExposureFrame)
    def __init__(self, exposure_change: ExposureChange, max_offset: int, min_offset: int=5, ratio_err: float = 0.015) -> None:
        self.exposure_change = exposure_change
        self.max_offset = max_offset
        self.min_offset = min_offset
        self.checked_offsets = set()
        self._are_all_offsets_exhausted = False
        self.ratio = IntensityDiffToExposureDiffRatio.init_of_exposure(self.exposure_change.prev_frame.exposure, err=ratio_err)
    
    def are_valid_intensity_frames(self, cur_frame_id: int, prev_frame_id: int) -> bool:
        return 0 < cur_frame_id - prev_frame_id
    
    def is_valid_offset(self, offset: int) -> bool:
        return self.min_offset <= offset <= self.max_offset
             
    def is_intensity_diff_matches_estimation(self, intensity_diff) -> bool:
        expected_intensity_diff = self.ratio.value * self.exposure_change.exposure_difference
        expected_intensity_diff_err =  abs(self.ratio.error * self.exposure_change.exposure_difference)
        abs_diff = abs(expected_intensity_diff - intensity_diff) 
        return abs_diff <= expected_intensity_diff_err

    def validate_offset(self, cur_frame: IntensityExposureFrame, prev_frame: IntensityExposureFrame) -> bool:
        if cur_frame is None or prev_frame is None:
            return False
        offset = cur_frame.id - self.exposure_change.id
        self.update_are_all_offsets_exhausted(offset)
        if not self.is_valid_offset(offset):
            return False
        if offset in self.checked_offsets:
            return False
        self.checked_offsets.add(offset)
        return True
        

    def validate_intensity(self, cur_frame: IntensityExposureFrame, prev_frame: IntensityExposureFrame) -> bool:
        if not self.are_valid_intensity_frames(cur_frame.id, prev_frame.id):
            return False
        intensity_diff = (cur_frame.intensity - prev_frame.intensity) / (cur_frame.id - prev_frame.id)
        return self.is_intensity_diff_matches_estimation(intensity_diff)
        
    @property
    def are_all_offsets_exhausted(self) -> bool:
        return self._are_all_offsets_exhausted
    
    def update_are_all_offsets_exhausted(self, offset:int):
        if not self._are_all_offsets_exhausted:
            self._are_all_offsets_exhausted = offset >= self.max_offset
    
    @property
    def are_all_offsets_tested(self):
        return len(self.checked_offsets) == self.max_offset - self.min_offset + 1

class ExposureTimeChangeDetector: 
    def __init__(self, max_offset: int = 20, ratio_err: float = 0.015):
        self.max_offset = max_offset
        self.cur_frame = None
        self.last_intensity_frame = None
        self.last_exposure_frame = None
        self.changes_validations_buffer = []
        self.max_missing_exposure_frames = 1
        self.ratio_err = ratio_err

    def is_valid_exposure_diff(self) -> bool:
        if self.last_exposure_frame is None:
            return False
        return self.cur_frame.id - self.last_exposure_frame.id <= self.max_missing_exposure_frames + 1

    def calc_exposure_diff(self) -> float:
        if self.is_valid_exposure_diff():
           return self.cur_frame.exposure - self.last_exposure_frame.exposure
        return np.nan

    def validate_change(self) -> Optional[ExposureChangeDetectionResult]:
        if len(self.changes_validations_buffer) > 1:
            for exposure_change, exposure_change_validation in self.changes_validations_buffer[1:]:
                is_valid_offset = exposure_change_validation.validate_offset(self.cur_frame, self.last_intensity_frame) 
        if len(self.changes_validations_buffer) > 0:
            exposure_change, exposure_change_validation = self.changes_validations_buffer[0]
            is_valid_offset = exposure_change_validation.validate_offset(self.cur_frame, self.last_intensity_frame) 
            is_valid_intensity = exposure_change_validation.validate_intensity(self.cur_frame, self.last_intensity_frame) 
            is_matching = is_valid_offset and is_valid_intensity
            if is_matching:
                self.changes_validations_buffer.pop(0)
                return ExposureChangeDetectionResult(change=exposure_change, 
                                                    status=ExposureValidationStatus.SUCCESS,
                                                    matching=self.cur_frame)
            if exposure_change_validation.are_all_offsets_exhausted:
                self.changes_validations_buffer.pop(0)
                if exposure_change_validation.are_all_offsets_tested:
                    return ExposureChangeDetectionResult(change=exposure_change, 
                                                    status=ExposureValidationStatus.FAIL)
                return ExposureChangeDetectionResult(change=exposure_change, 
                                                    status=ExposureValidationStatus.INCOMPLETE)
            return ExposureChangeDetectionResult(change=exposure_change, 
                                                status=ExposureValidationStatus.EVALUATION)
            
    def feed_frame(self, frame_id: int, exposure: float, intensity: float) -> ExposureChangeDetectionResult:
        self.pre_process(frame_id, exposure, intensity)
        detection_result = self.validate_change()
        self.post_process()
        return detection_result

    def pre_process(self, frame_id: int, exposure: float, intensity: float) -> None:
        self.cur_frame = IntensityExposureFrame(frame_id, exposure, intensity)
        exposure_difference = self.calc_exposure_diff()
        if not np.isnan(exposure_difference) and exposure_difference != 0:
            exposure_change = ExposureChange(cur_frame=self.cur_frame,prev_frame=self.last_exposure_frame)
            exposure_change_validator = ExposureChangeValidator(exposure_change, self.max_offset, ratio_err=self.ratio_err)
            self.changes_validations_buffer.append((exposure_change, exposure_change_validator))

    def update_last_exposure_frame(self) -> None:
        if not np.isnan(self.cur_frame.exposure):
            self.last_exposure_frame = self.cur_frame
    
    def update_last_intensity_frame(self) -> None:
        if not np.isnan(self.cur_frame.intensity):
            self.last_intensity_frame = self.cur_frame
    
    def post_process(self) -> None:
        self.update_last_exposure_frame()
        self.update_last_intensity_frame()

if __name__ == "__main__":
    import adaptive_parameters.utils as utils
    from pathlib import Path
    
    adaptive_parameters_path = Path('INPUT\\10_8_23\\adaptive_parameters_2023_08_10_15_31_21.json')
    frames_dir = Path('OUTPUT\\10_8_23\\recording_2023_08_10_15_31_21_images')
    
    frames_exposure_id, frames_exposure_time = utils.read_exposure_data(adaptive_parameters_path)
    exposure_df = pd.DataFrame({'exposure': frames_exposure_time}, index=frames_exposure_id)
    frames_intensity_id, frames_intensity = utils.read_intensity_data(frames_dir)
    intensity_df = pd.DataFrame({'intensity': frames_intensity}, index=frames_intensity_id)
    exposure_intensity_df = pd.merge(exposure_df, intensity_df, how="outer", left_index=True, right_index=True)
    
    ratio_err = 0.015

    detector = ExposureTimeChangeDetector(ratio_err=ratio_err)
    changes_detected = {}
    for frame_id, frame_data in exposure_intensity_df[3:].iterrows():
        detection_result = detector.feed_frame(frame_id, frame_data['exposure'], frame_data['intensity'])
        if detection_result is not None and detection_result.status != ExposureValidationStatus.EVALUATION:
            changes_detected[detection_result.change.id] = detection_result
    
    print(f'# detected = {len([k for k,v in changes_detected.items() if v.status==ExposureValidationStatus.SUCCESS])}')
    print(f'# not detected = {len([k for k,v in changes_detected.items() if v.status==ExposureValidationStatus.FAIL])}')
    print(f'# not completed = {len([k for k,v in changes_detected.items() if v.status==ExposureValidationStatus.INCOMPLETE])}')
    print('---------------------------------------------')
    print(f'# detected = {len([k for k,v in changes_detected.items() if v.status==ExposureValidationStatus.SUCCESS])}')
    print(f'# not detected + # not completed = {len([k for k,v in changes_detected.items() if v.status==ExposureValidationStatus.FAIL or v.status==ExposureValidationStatus.INCOMPLETE])}')
    print('---------------------------------------------')
    print(f'# total = {len(list(changes_detected.items()))}')
    
    print('finished')