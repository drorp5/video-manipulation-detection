from __future__ import annotations
from dataclasses import dataclass, field
import json
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

@dataclass
class Frame:
    """Simplified frame object with id, expoure time and intensity"""
    id: int
    exposure: float
    intensity: float

    def get_exposure_diff(self, prev_frame:Frame):
        if self.id - prev_frame.id == 1:
           return self.exposure - prev_frame.exposure
        return np.nan

@dataclass
class ExposureChangeFrame:
    """Stores data of frame in which there was an exposure time change"""
    frame: Frame
    exposure_difference: float
    prev_exposure: float
    checked_offsets: set = field(default_factory=set) 
    checked_intensities_offsets: set = field(default_factory=set) 

    @property
    def frame_id(self):
        return self.frame.id
    

@dataclass
class FiniteFramesBuffer():
    """Stores limited number of frames"""
    size: int
    frames_buffer: list = field(default_factory=list) 

    def insert_frame(self, frame: Frame):
        if len(self.frames_buffer) == self.size:
            self.frames_buffer.pop(0)
        self.frames_buffer.append(frame)
    
    @property
    def num_frames(self) -> int:
        return len(self.frames_buffer)
    
    def get_last(self) -> Frame:
        if self.num_frames == 0:
            return None
        return self.frames_buffer[-1]
    
    def remove_till_index(self, ind: int) -> None:
        self.frames_buffer = self.frames_buffer[ind+1:]


class ExposureIntensityChangeValidator():
    # max_offset = 20
    # min_offset = 5

    # low_intensity_ratio = 0.05
    # low_intensity_ratio_err = 0.005
    
    # medium_intensity_ratio_err = 0.015

    # high_intensity_ratio = 0.0067
    # high_intensity_ratio_err = 0.0017
    
    # low_intensity = 110
    # high_intensity = 200
    def __init__(self, max_offset: int) -> None:
        df_path = r'./INPUT/exposure_intensity_ratio.csv'
        self.ratio_df = pd.read_csv(df_path)
        self.max_offset = max_offset
        self.min_offset = 5



    # def get_exposure_diff_ratio(self, intensity: float) -> Tuple[float, float]:
    #     if intensity < self.low_intensity:
    #         ratio = self.low_intensity_ratio
    #         ratio_err = self.low_intensity_ratio_err
    #     elif intensity > self.high_intensity:
    #         ratio = self.high_intensity_ratio
    #         ratio_err = self.high_intensity_ratio_err
    #     else:
    #         ratio = (self.high_intensity_ratio - self.low_intensity_ratio)/(self.high_intensity - self.low_intensity) * (intensity - self.low_intensity) + self.low_intensity_ratio
    #         ratio_err = self.medium_intensity_ratio_err

    #     return ratio, ratio_err    


    def get_exposure_diff_ratio(self, exposure: float) -> Tuple[float, float]:
        ind = np.argmin(abs(self.ratio_df['exposure'] - exposure))
        err =  self.ratio_df['ratio_std'][ind]
        if exposure < 2500:
            err = max(err, 0.01)
        return self.ratio_df['ratio'][ind], err


    def validate(self, exposure_change_frame: ExposureChangeFrame, cur_frame: Frame, prev_frame: Frame):
        if cur_frame.id - prev_frame.id != 1:
            return False
        offset = cur_frame.id - exposure_change_frame.frame_id 
        if offset < self.min_offset:
            return False
        if offset > self.max_offset:
            return False
        if offset in exposure_change_frame.checked_offsets:
            return False
        exposure_change_frame.checked_offsets.add(offset)
        # match intensity diff
        intensity_diff = cur_frame.intensity - prev_frame.intensity
        if np.isnan(intensity_diff):
            return False
        exposure_change_frame.checked_intensities_offsets.add(offset)
        # ratio, ratio_err = self.get_exposure_diff_ratio(prev_frame.intensity)
        ratio, ratio_err = self.get_exposure_diff_ratio(exposure_change_frame.prev_exposure)
        expected_intensity_diff = ratio * exposure_change_frame.exposure_difference
        expected_intensity_diff_err = abs(ratio_err * exposure_change_frame.exposure_difference)
        abs_diff = abs(expected_intensity_diff - intensity_diff) 
        is_diff_match = abs_diff < expected_intensity_diff_err
        return is_diff_match
        
    def all_options_exhausted(self, exposure_change_frame: ExposureChangeFrame, frames_buffer: FiniteFramesBuffer):
        return frames_buffer.get_last().id - exposure_change_frame.frame_id > self.max_offset

    def all_options_tested(self, exposure_change_frame:ExposureChangeFrame):
        return len(exposure_change_frame.checked_intensities_offsets) == self.max_offset - self.min_offset + 1

class FrameDataBuffer:
    def __init__(self):
        self.max_offset = 20
        self.frames_buffer = FiniteFramesBuffer(self.max_offset)
        self.changes_buffer = []
        self.exposure_change_validator = ExposureIntensityChangeValidator(self.max_offset)

    def add_frame_data(self, frame_id: int, exposure: float, intensity: float):
        cur_frame = Frame(frame_id, exposure, intensity)
        prev_frame = self.frames_buffer.get_last()
        
        if prev_frame is not None:
            exposure_difference = cur_frame.get_exposure_diff(prev_frame)
            if not np.isnan(exposure_difference) and exposure_difference != 0:                
                self.changes_buffer.append(ExposureChangeFrame(frame=cur_frame, exposure_difference=exposure_difference, prev_exposure=prev_frame.exposure))

        self.frames_buffer.insert_frame(cur_frame)
    
    def check_changes(self) -> Tuple[ExposureChangeFrame, Frame]:
        matching_frame = None
        if len(self.changes_buffer) > 0 and self.frames_buffer.num_frames > 1:
            change = self.changes_buffer[0]
            for frame_ind, (prev_frame, cur_frame) in enumerate(zip(self.frames_buffer.frames_buffer[:-1], self.frames_buffer.frames_buffer[1:])):
                matching = self.exposure_change_validator.validate(change, cur_frame, prev_frame)
                if matching:
                    matching_frame = cur_frame
                    break
            if matching:
                self.frames_buffer.remove_till_index(frame_ind)
                self.changes_buffer.pop(0)
                return change, matching_frame
            elif self.exposure_change_validator.all_options_exhausted(change, self.frames_buffer):
                self.changes_buffer.pop(0)
                if self.exposure_change_validator.all_options_tested(change):
                    return change, None
            return None
       
    

if __name__ == "__main__":
    # with open (r"./INPUT/adaptive_parameters_15_31_45.json", 'r') as f:
    #     data = json.load(f)
    # frames_exposure_id = []
    # frames_exposure_time = []
    # for frame_id, frame_data in data.items():
    #     try:
    #         frames_exposure_id.append(int(frame_id.split('_')[1]))
    #         frames_exposure_time.append(frame_data["exposure_us"])
    #     except:
    #         continue

    # frames_exposure_id = np.array(frames_exposure_id)
    # frames_exposure_time = np.array(frames_exposure_time)

    # with open(r"./OUTPUT/recording_15_31_45_images/averaged_intensities.txt", 'r') as f:
    #     intensities_txt = f.read()
    # result = re.findall(r"frame (\d+): (\d+\.\d+)", intensities_txt)

    # frames_id = []
    # frames_intensity = []
    # for frame_id, intensity in result:
    #     frames_id.append(int(frame_id))
    #     frames_intensity.append(float(intensity))

    # frames_id = np.array(frames_id)
    # frames_intensity = np.array(frames_intensity)
    # interpolated_values = np.interp(frames_id, frames_exposure_id, frames_exposure_time)

    # intensity_df = pd.DataFrame({'intensity': frames_intensity}, index=frames_id)
    # exposure_df = pd.DataFrame({'exposure': frames_exposure_time}, index=frames_exposure_id)
    # df = pd.merge(exposure_df, intensity_df, how="outer", left_index=True, right_index=True)
    

    # buffer = FrameDataBuffer()
    # changes_detected = {}
    # for frame_id, frame_data in df[3:].iterrows():
    #     buffer.add_frame_data(frame_id, frame_data['exposure'], frame_data['intensity'])
    #     res = buffer.check_changes()
    #     while res is not None:
    #         exposure_change, matching_frame = res
    #         if matching_frame is not None:
    #             changes_detected[exposure_change.frame_id] = matching_frame.id
    #         else:
    #             changes_detected[exposure_change.frame_id] = None
    #         res = buffer.check_changes()
    
    # df2 = pd.read_csv(r".\OUTPUT\recording_15_31_45_images\adaptive_parameters_15_31_45_exposure_intensity_matches.txt")
    # # set(df2['exposure_frame'])
    # detected = []
    # not_detected = []
    # for k,v in changes_detected.items():
    #     if k > 3000:
    #         break
    #     if v is None:
    #         not_detected.append(k)
    #     else:
    #         detected.append(k)

    
    with open ("INPUT/adaptive_parameters_09_44_53.json", 'r') as f:
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

    with open(r"./OUTPUT/recording_09_44_53_images/averaged_intensities.txt", 'r') as f:
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
    df2 = pd.merge(exposure_df, intensity_df, how="outer", left_index=True, right_index=True)
    # df = df.fillna(method='ffill')

    buffer = FrameDataBuffer()
    changes_detected = {}
    for frame_id, frame_data in df2[3:].iterrows():
        buffer.add_frame_data(frame_id, frame_data['exposure'], frame_data['intensity'])

        res = buffer.check_changes()
        while res is not None:
            exposure_change, matching_frame = res
            if matching_frame is not None:
                changes_detected[exposure_change.frame_id] = matching_frame.id
            else:
                changes_detected[exposure_change.frame_id] = None
            res = buffer.check_changes()

        
    print('finished')