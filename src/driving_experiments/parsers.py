from typing import Tuple
from pathlib import Path
import re
import pandas as pd


def get_logged_frames_info(log_path: Path) -> Tuple[int, int, int]:
    if not log_path.exists():
        raise FileNotFoundError

    unique_frames = set()
    frame_id_regex = r"Frame # (\d+)"
    with open(log_path.as_posix(), "r") as log_file:
        for line in log_file:
            matches = re.findall(frame_id_regex, line)
            for match in matches:
                unique_frames.add(int(match))
    if len(unique_frames) == 0:
        return 0, 0, 0
    return min(unique_frames), max(unique_frames), len(unique_frames)


def get_number_of_detections_frames(log_path: Path) -> int:
    if not log_path.exists():
        raise FileNotFoundError
    detections = 0
    with open(log_path.as_posix(), "r") as log_file:
        for line in log_file:
            if "DETECTIONS" in line:
                detections += 1
    return detections


def summarize_log_file(log_path: Path) -> str:
    try:
        first_frame, last_frame, num_frames = get_logged_frames_info(log_path)
        detections_frames = get_number_of_detections_frames(log_path)
    except FileNotFoundError:
        return "Log Not Found"

    summary = f"First frame ID = {first_frame}\n"
    summary = f"Last frame ID = {last_frame}\n"
    summary += f"# Logged Frames = {num_frames}\n"
    summary += f"Total {100*num_frames/(last_frame-first_frame+1):.1f}% logged\n"
    summary += f"Detections Frames = {detections_frames}"
    return summary


def extract_frame_width_data(log_path: Path):
    """
    This function reads a log file and extracts frame number, transmitted width, and received value using regular expressions.

    Args:
        log_path: Path to the log file.

    Returns:
        A pandas DataFrame containing frame number, transmitted width, and received value.
    """

    data = []
    transmitted_pattern = "Frame # (\d+): Setting next frame width = (\d+)"
    received_pattern = "Frame # (\d+): (\d+) -> (\w+)"

    prev_frame_num = None
    with open(log_path.as_posix(), "r") as f:
        for line in f:
            matched_transmitted = re.search(transmitted_pattern, line)
            match_received = re.search(received_pattern, line)
            if matched_transmitted:
                prev_frame_num, trasmitted_value = matched_transmitted.groups()
                prev_frame_num = int(prev_frame_num)
                trasmitted_value = int(trasmitted_value)
            elif match_received:
                frame_num, received_value, result = match_received.groups()
                frame_num = int(frame_num)
                received_value = int(received_value)

                if frame_num == prev_frame_num + 1:
                    data.append(
                        {
                            "frame_number": frame_num,
                            "transmitted": trasmitted_value,
                            "received": received_value,
                            "result": result,
                        }
                    )

    # Create pandas DataFrame from the extracted data
    df = pd.DataFrame(data)
    return df
