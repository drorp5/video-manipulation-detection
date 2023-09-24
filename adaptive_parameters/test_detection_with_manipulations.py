import random
import pandas as pd
import adaptive_parameters.utils as utils
from pathlib import Path
import cv2
import numpy as np
from adaptive_parameters.exposure_time_detection import ExposureTimeChangeDetector, ExposureValidationStatus

random.seed(42)

class ExposureAnomalyDetectionTester():
    def __init__(self, exposure_change_detector, manipulation_probability: float, manipulation_intensity: float) -> None:
        self.exposure_change_detector = exposure_change_detector
        self.min_offset = exposure_change_detector.min_offset
        self.max_offset = exposure_change_detector.max_offset
        self.ratio_err = exposure_change_detector.ratio_err
        self.manipulation_probability = manipulation_probability
        self.manipulation_intensity = manipulation_intensity 

    def manipulate_constant_intensity(self, exposure_intensity_df: pd.DataFrame):
        nan_filled_exposure = utils.fill_first_nan_value(exposure_intensity_df['exposure'])
        change_frame_ids = utils.get_not_nan_change_indices(nan_filled_exposure)
        manipulation_df = exposure_intensity_df.copy()
        manipulated_frames = []
        last_manipulated_frame = None
        for frame_id in change_frame_ids:
            if last_manipulated_frame is not None and (frame_id - last_manipulated_frame) < (self.max_offset - self.min_offset):
                continue
            random_variable = random.uniform(0, 1)
            if random_variable < self.manipulation_probability:
                possible_ids = manipulation_df.index[(manipulation_df.index >= frame_id + self.min_offset) & (manipulation_df.index <= frame_id + self.max_offset)].tolist()
                for manipulated_id in possible_ids:
                # if len(possible_ids) > 0:
                    # manipulated_id = random.choice(possible_ids)
                    manipulation_df.loc[manipulated_id]['intensity'] = self.manipulation_intensity
                    manipulated_frames.append(manipulated_id)
                    last_manipulated_frame = manipulated_id
        return manipulation_df, manipulated_frames

    def detect_anomalies(self, exposure_intensity_df: pd.DataFrame):
        changes_detected = utils.detect_changes(exposure_intensity_df, self.exposure_change_detector)
        anomalies_detected = {}
        last_left_id = None
        last_right_id = None

        changes_detected_sorted = sorted(changes_detected.items(), key=lambda x: x[0]) 

        for id, detection_result in changes_detected_sorted:
            manipulation_detected = detection_result.status!=ExposureValidationStatus.SUCCESS
            if last_right_id is not None and id + self.min_offset < last_right_id:
                prev_manipulation_detected = anomalies_detected.pop((last_left_id, last_right_id))
                last_right_id = id + self.max_offset
                manipulation_detected = manipulation_detected or prev_manipulation_detected
            else:
                last_left_id = id + self.min_offset
                last_right_id = id + self.max_offset
            anomalies_detected[(last_left_id, last_right_id)] = manipulation_detected
        return anomalies_detected

    def manipulate_and_detect(self, exposure_intensity_df: pd.DataFrame):
        manipulated_exposure_intensity_df, manipulated_frames = self.manipulate_constant_intensity(exposure_intensity_df)
        events = self.detect_anomalies(manipulated_exposure_intensity_df)

        assert all([any([left<=manipulated_frame<=right for left,right in events.keys()]) for manipulated_frame in manipulated_frames])

        #TODO: faster
        segments = []
        pred = []
        label = []
        for segment_bounds, anomaly_detected in events.items():
            segments.append(segment_bounds)
            pred.append(anomaly_detected)
            anomaly_in_segment = False
            for manipulated_frame in manipulated_frames:
                if segment_bounds[0] <= manipulated_frame <= segment_bounds[1]:
                    #TODO: can delete if found
                    anomaly_in_segment = True
                    break
            label.append(anomaly_in_segment)
        return segments, pred, label

if __name__ == "__main__":
    # adaptive_parameters_path = Path(r'INPUT\10_8_23\adaptive_parameters_2023_08_10_15_31_21.json')
    # frames_dir = Path(r'OUTPUT\10_8_23\recording_2023_08_10_15_31_21_images')
    adaptive_parameters_path = Path(r"INPUT/8_8_23/adaptive_parameters_10_18_48.json")
    frames_dir = Path(r"./OUTPUT/8_8_23/recording_10_18_48_images/")
    fake_image = Path(r'INPUT\stop_sign_road.jpg')

    frames_exposure_id, frames_exposure_time = utils.read_exposure_data(adaptive_parameters_path)
    exposure_df = pd.DataFrame({'exposure': frames_exposure_time}, index=frames_exposure_id)
    frames_intensity_id, frames_intensity = utils.read_intensity_data(frames_dir)
    intensity_df = pd.DataFrame({'intensity': frames_intensity}, index=frames_intensity_id)
    exposure_intensity_df = pd.merge(exposure_df, intensity_df, how="outer", left_index=True, right_index=True)
    
    min_offset = 5
    max_offset = 20
    ratio_err = 0.01
    manipulation_probability = 0.5
    manipulation_intensity = np.mean(cv2.cvtColor(cv2.imread(fake_image.as_posix()), cv2.COLOR_BGR2GRAY).astype(float))
    
    detector = ExposureTimeChangeDetector(ratio_err=ratio_err)

    tester = ExposureAnomalyDetectionTester(detector, manipulation_probability=manipulation_probability, manipulation_intensity=manipulation_intensity)
    segments, pred, label = tester.manipulate_and_detect(exposure_intensity_df)

    print('finished')
    
    