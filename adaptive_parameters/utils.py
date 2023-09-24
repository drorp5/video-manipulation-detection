import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .exposure_time_detection import ExposureTimeChangeDetector, ExposureChangeDetectionResult, ExposureValidationStatus
import random

def read_exposure_data(adaptive_parameters_path: Path):
    with open (adaptive_parameters_path, 'r') as f:
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
    return frames_exposure_id, frames_exposure_time

def read_intensity_data(frames_dir: Path):
    frame_intensity_path = frames_dir / f"averaged_intensities.txt"
    with open(frame_intensity_path, 'r') as f:
        intensities_txt = f.read()
    result = re.findall(r"frame (\d+): (\d+\.\d+)", intensities_txt)

    frames_intensity_id = []
    frames_intensity = []
    for frame_id, intensity in result:
        frames_intensity_id.append(int(frame_id))
        frames_intensity.append(float(intensity))

    frames_intensity_id = np.array(frames_intensity_id)
    frames_intensity = np.array(frames_intensity)
    return frames_intensity_id, frames_intensity

def read_matches_data(matches_path: Path, exposure_df: pd.DataFrame, intensity_df: pd.DataFrame) ->pd.DataFrame:
    df = pd.read_csv(matches_path.as_posix())
    frames_exposure_id = exposure_df.index.to_numpy()
    frames_exposure_time = exposure_df['exposure'].to_numpy()
    frames_intensity_id = intensity_df.index.to_numpy()
    frames_intensity_value = intensity_df['intensity'].to_numpy()

    # manual_exposure_time_change_frames_inds =  np.searchsorted(frames_exposure_id, df['exposure_frame'])
    # df['exposure_diff'] = frames_exposure_time[manual_exposure_time_change_frames_inds] - frames_exposure_time[manual_exposure_time_change_frames_inds-1]

    non_nan_indices = np.argwhere(~np.isnan(df['intensity_frame']))[:,0]
    manual_intensity_change_frames_id_filtered = df['intensity_frame'][non_nan_indices]

    manual_intensity_change_frames_inds =  np.searchsorted(frames_intensity_id, manual_intensity_change_frames_id_filtered)
    manual_intensity_change_frames_values_filtered = frames_intensity_value[manual_intensity_change_frames_inds] - frames_intensity_value[manual_intensity_change_frames_inds-1]

    manual_intensity_change_frames_values = np.zeros(len(df))
    manual_intensity_change_frames_values[non_nan_indices] = manual_intensity_change_frames_values_filtered
    manual_intensity_change_frames_values[manual_intensity_change_frames_values==0] = np.nan
    df['intensity_diff'] = manual_intensity_change_frames_values

    df['ratio'] = df['intensity_diff'] / df['exposure_diff']
    df['offset'] = df['intensity_frame'] - df['exposure_frame']
    df['intensity'] = frames_intensity_value[np.searchsorted(frames_intensity_id, df['intensity_frame'])-1]
    df['exposure'] = frames_exposure_time[np.searchsorted(frames_exposure_id, df['exposure_frame'])-1]
    return df

def detect_changes(exposure_intensity_df, detector: ExposureTimeChangeDetector):
    changes_detected = {}
    for frame_id, frame_data in exposure_intensity_df.iterrows():
        detection_result = detector.feed_frame(frame_id, frame_data['exposure'], frame_data['intensity'])
        if detection_result is not None and detection_result.status != ExposureValidationStatus.EVALUATION:
            changes_detected[detection_result.change.id] = detection_result
    return changes_detected

def set_detection_matches(manual_matches_df, changes_detected):
    manual_changes_1_df = manual_matches_df.copy()
    manual_changes_1_df['abs_exposure_diff'] = abs(manual_changes_1_df['exposure_diff'])
    manual_changes_1_df['detected'] = [changes_detected[k].status.name if k in changes_detected else 'NOT OBSERVED' for k in manual_changes_1_df['exposure_frame']]
    matching_frame = []
    for k in manual_changes_1_df['exposure_frame']:
        if k not in changes_detected:
            matching_frame.append(None)
        elif changes_detected[k].status == ExposureValidationStatus.SUCCESS:
            matching_frame.append(changes_detected[k].matching.id)
        else:
            matching_frame.append(None)
    manual_changes_1_df['matching_frame'] = matching_frame
    manual_changes_1_df['matching_diff'] = manual_changes_1_df['matching_frame'] - manual_changes_1_df['intensity_frame']
    return manual_changes_1_df 

def fill_first_nan_value(series: pd.Series) -> pd.Series:
    # Fill only the first NaN value within consecutive NaN values
    series_copy = series.copy()
    for change_index in range(1, len(series_copy)):
        if pd.isna(series_copy.iloc[change_index]):
            if series_copy.index[change_index] - series_copy.index[change_index-1] == 1:
                series_copy.iloc[change_index] = series.iloc[change_index-1]
    return series_copy

def get_not_nan_change_indices(series:pd.Series) -> pd.Index:
    ids_mask = np.diff(series.index) == 1
    ids_mask = np.insert(ids_mask, 0, True)
    values_mask = series.ne(series.shift()) & ~series.isna() & ~series.shift().isna()
    mask = ids_mask & values_mask
    return series.index[mask]