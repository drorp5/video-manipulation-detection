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
        self.ratio_df = pd.read_csv(r'./INPUT/exposure_intensity_ratio.csv') #TODO: change this to estimated function
        self.max_offset = max_offset
        self.min_offset = min_offset
        self.checked_offsets = set()
        self.max_missing_intensity_frames = 0
        self._are_all_options_exhausted = False

    def estimate_intensity_diff_to_exposure_diff_ratio(self, exposure: float) -> Tuple[float, float]:
        ind = np.argmin(abs(self.ratio_df['exposure'] - exposure))
        err =  self.ratio_df['ratio_std'][ind]
        if exposure < 2500: #TODO: change this according to the estimated function
            err = max(err, 0.01)
        return self.ratio_df['ratio'][ind], err

    def are_valid_intensity_frames(self, cur_frame_id: int, prev_frame_id) -> bool:
        return 0 < cur_frame_id - prev_frame_id <= self.max_missing_intensity_frames + 1

    def is_valid_offset(self, offset: int) -> bool:
        return self.min_offset <= offset <= self.max_offset
    
    def is_valid_intensity_diff(self, intensity_diff):
        return not np.isnan(intensity_diff)
             
    def is_intensity_diff_matches_estimation(self, intensity_diff) -> bool:
        ratio, ratio_err = self.estimate_intensity_diff_to_exposure_diff_ratio(self.exposure_change.prev_frame.exposure)
        expected_intensity_diff = ratio * self.exposure_change.exposure_difference
        expected_intensity_diff_err = abs(ratio_err * self.exposure_change.exposure_difference)
        abs_diff = abs(expected_intensity_diff - intensity_diff) 
        return abs_diff <= expected_intensity_diff_err

    def validate(self, cur_frame: IntensityExposureFrame, prev_frame: IntensityExposureFrame) -> bool:
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
                                                      status=ExposureValidationStatus.FAIL,)
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
    with open ("INPUT/8_8_23/adaptive_parameters_10_08_18.json", 'r') as f:
        data = json.load(f)
    frames_exposure_id = []
    frames_exposure_time = []
    for frame_id, frame_data in data.items():
        try:
            frames_exposure_id.append(int(frame_id.split('_')[1]))
            frames_exposure_time.append(frame_data["exposure_us"])
        except:
            continue

    frames_exposure_id = np.array(frames_exposure_id)
    frames_exposure_time = np.array(frames_exposure_time)

    with open(r"./OUTPUT/8_8_23/recording_10_08_18_images/averaged_intensities.txt", 'r') as f:
        intensities_txt = f.read()
    result = re.findall(r"frame (\d+): (\d+\.\d+)", intensities_txt)

    frames_id = []
    frames_intensity = []
    for frame_id, intensity in result:
        frames_id.append(int(frame_id))
        frames_intensity.append(float(intensity))

    frames_id = np.array(frames_id)
    frames_intensity = np.array(frames_intensity)
    interpolated_values = np.interp(frames_id, frames_exposure_id, frames_exposure_time)

    intensity_df = pd.DataFrame({'intensity': frames_intensity}, index=frames_id)
    exposure_df = pd.DataFrame({'exposure': frames_exposure_time}, index=frames_exposure_id)
    exposure_intensity_df = pd.merge(exposure_df, intensity_df, how="outer", left_index=True, right_index=True)

    detector = ExposureTimeChangeDetector()
    changes_detected = {}
    for frame_id, frame_data in exposure_intensity_df[3:].iterrows():
        detection_result = detector.feed_frame(frame_id, frame_data['exposure'], frame_data['intensity'])
        if detection_result is not None and detection_result.status != ExposureValidationStatus.EVALUATION:
            changes_detected[detection_result.change.id] = detection_result
        
        
    print('finished')