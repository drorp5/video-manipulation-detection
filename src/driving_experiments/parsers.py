from typing import Dict, List, Tuple
from pathlib import Path
import re
import pandas as pd
import ast

from active_manipulation_detectors.validation_status import ValidationStatus


import re
from datetime import datetime

def parse_log_file(file_path: Path) -> pd.DataFrame:
    with open(file_path.as_posix(), 'r') as file:
        log_lines = file.readlines()
    
    log_pattern = r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<level>\w+) - (?P<message>.*)$'
    parsed_data = []
    for line in log_lines:
        # Extracting the timestamp, log level, and message
        match = re.match(log_pattern, line)
        if match:
            log_data = match.groupdict()
            timestamp  = datetime.strptime(log_data['timestamp'], '%Y-%m-%d %H:%M:%S,%f')
            # Parsing INFO messages related to frames
            if log_data['level'] == 'INFO':
                frame_data = {'timestamp': timestamp}
                frame_pattern = r'\{(.*?)\}'
                frame_match = re.search(frame_pattern, log_data['message'])
                if frame_match:
                    frame_data_str = frame_match.group(1)
                    
                    # Extracting key-value pairs from the frame data
                    key_value_pattern = r"\'(\w+)\': (\d+|\'[^\']*\'|\[\([\d, ]+\)\])"
                    for kv_match in re.findall(key_value_pattern, frame_data_str):
                        key, value = kv_match
                        frame_data[key] = ast.literal_eval(value)
                        
                                
                parsed_data.append(frame_data)
    
    return pd.DataFrame(parsed_data)


def get_validation_results_summary(log_path: Path) -> Dict[str, int]:
    if not log_path.exists():
        raise FileNotFoundError
    res = {name: 0 for name in ValidationStatus._member_names_}
    received_pattern = "Frame # \d+: \d+ -> (\w+)"
    with open(log_path.as_posix(), "r") as log_file:
        for line in log_file:
            match_received = re.search(received_pattern, line)
            if match_received:
                result = match_received.groups()[0]
                res[result] += 1
    return res


def summarize_log_file(log_path: Path) -> str:
    try:
        frames_log_df = parse_log_file(log_path)
        first_frame = frames_log_df["frame_id"].iloc[0]
        last_frame = frames_log_df["frame_id"].iloc[-1]
        num_frames = len(frames_log_df)
        if "detections" in frames_log_df:
            detections_frames = frames_log_df["detections"].count()
        else:
            detections_frames = 0
        validation_summary = dict(frames_log_df['validation_result'].value_counts())
    except FileNotFoundError:
        return "Log Not Found"

    summary = f"First frame ID = {first_frame}\n"
    summary = f"Last frame ID = {last_frame}\n"
    summary += f"# Logged Frames = {num_frames}\n"
    summary += f"# Total {100*num_frames/(last_frame-first_frame+1):.1f}% logged\n"
    summary += f"# Detections Frames = {detections_frames}\n"
    summary += f"# Validation: {validation_summary}"
    return summary


def extract_frame_width_data(log_path: Path):
    """
    Args:
        log_path: Path to the log file.

    Returns:
        A pandas DataFrame containing frame number, transmitted width, and received value.
    """

    frames_df = parse_log_file(log_path)
    res_df = frames_df.loc[:, ["frame_id", "width", "next_width", "validation_result"]]
    res_df_off_one = res_df.loc[1:, ["frame_id", "width", "validation_result"]]
    res_df_off_one["transmitted"] = list(frames_df["next_width"][:-1])
    res_df_off_one["transmitted"] = res_df_off_one["transmitted"].astype(int)
    res_df_off_one["width"] = res_df_off_one["width"].astype(int)
    res_df_off_one.rename(columns={"width": "received"}, inplace=True)
    return res_df_off_one
    

if __name__ == "__main__":
    log_file = Path(r"C:\Users\user\Desktop\Dror\video-manipulation-detection\OUTPUT\tmp\2024_06_11_17_03_39_8022a5c8-baa4-4c7e-a1b4-f66b13c0de02\log_8022a5c8-baa4-4c7e-a1b4-f66b13c0de02.log")
    res = summarize_log_file(log_file)
    print(res)