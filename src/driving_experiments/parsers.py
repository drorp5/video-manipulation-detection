from typing import Dict, List, Tuple
from pathlib import Path
import re
import pandas as pd
import ast
from datetime import datetime
import cv2

from active_manipulation_detectors.validation_status import ValidationStatus
from gige.attacked_gvsp_parser import AttackedGvspPcapParser
from gige.gvsp_frame import gvsp_frame_to_rgb


def parse_log_file(file_path: Path) -> pd.DataFrame:
    with open(file_path.as_posix(), "r") as file:
        log_lines = file.readlines()

    log_pattern = r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<level>\w+) - (?P<message>.*)$"
    parsed_data = []
    for line in log_lines:
        # Extracting the timestamp, log level, and message
        match = re.match(log_pattern, line)
        if match:
            log_data = match.groupdict()
            timestamp = datetime.strptime(log_data["timestamp"], "%Y-%m-%d %H:%M:%S,%f")
            # Parsing INFO messages related to frames
            if log_data["level"] == "INFO":
                frame_data = {"timestamp": timestamp}
                frame_pattern = r"\{(.*?)\}"
                frame_match = re.search(frame_pattern, log_data["message"])
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
        validation_summary = dict(frames_log_df["validation_result"].value_counts())
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


def extract_frames_of_pcap(pcap_path: Path, log_path: Path, dst_dir: Path) -> None:
    """
    Extracts specific frames from a pcap file based on timestamps provided in a log file and saves them to a destination directory.

    This function reads a pcap file and a log file containing timestamps, extracts the frames from the pcap file that match the timestamps in the log, and saves the extracted frames to the specified destination directory.

    Parameters:
    -----------
    pcap_path : Path
        The path to the pcap file from which frames will be extracted.

    log_path : Path
        The path to the log file containing timestamps of the frames to be extracted.

    dst_dir : Path
        The destination directory where the extracted frames will be saved.

    Returns:
    --------
    None
        This function does not return anything. It performs the extraction and saving of frames as a side effect.
    """

    pcap_parser = AttackedGvspPcapParser(pcap_path)
    frames_df = parse_log_file(log_path)
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)

    saved_frames = 0
    current_row = 0
    curernt_timestamp = frames_df["timestamp"].iloc[current_row]
    current_id = frames_df["frame_id"].iloc[current_row]
    for frame in pcap_parser.frames:
        while current_row < len(frames_df) - 1 and frame.timestamp > curernt_timestamp:
            current_row += 1
            curernt_timestamp = frames_df["timestamp"].iloc[current_row]
            current_id = frames_df["frame_id"].iloc[current_row]
        if frame.timestamp == curernt_timestamp:
            # save frame
            dst_path = dst_dir / f"frame_{current_id}_BlockID_{frame.id}.png"
            img = cv2.cvtColor(gvsp_frame_to_rgb(frame), cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path.as_posix(), img)
            saved_frames += 1

            current_row += 1
            if current_row < len(frames_df):
                curernt_timestamp = frames_df["timestamp"].iloc[current_row]
                current_id = frames_df["frame_id"].iloc[current_row]
    print(f"{saved_frames} frames saved")


def evaluate_success_recording_rate(log_path: Path) -> float:
    frames_log_df = parse_log_file(log_path)
    first_frame = frames_log_df["frame_id"].iloc[0]
    last_frame = frames_log_df["frame_id"].iloc[-1]
    num_frames = len(frames_log_df)
    return num_frames / (last_frame - first_frame + 1)


if __name__ == "__main__":
    log_path = Path(
        r"C:\Users\user\Desktop\Dror\video-manipulation-detection\OUTPUT\tmp\2024_06_12_10_46_11_24088764-7371-41b3-a243-348b39d8b078\log_24088764-7371-41b3-a243-348b39d8b078.log"
    )
    pcap_path = Path(
        r"C:\Users\user\Desktop\Dror\video-manipulation-detection\OUTPUT\tmp\2024_06_12_10_46_11_24088764-7371-41b3-a243-348b39d8b078\24088764-7371-41b3-a243-348b39d8b078.pcap"
    )
    dst_dir = log_path.parent / "pcap_frames"
    extract_frames_of_pcap(pcap_path, log_path, dst_dir)
