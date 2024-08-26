"""
This module provides functionality for summarizing GigE Vision experiment results.
It includes functions for extracting metadata from configuration files, processing log files,
and aggregating results across different experimental conditions.
"""

from typing import Any, Callable, List, Optional, Tuple
from tqdm import tqdm
import yaml
from pathlib import Path
from functools import partial

from attacker import Attackers
from active_detection_experiments.parsers import evaluate_success_recording_rate
from active_detection_experiments.tags import RoadType, TimeOfDay
from active_detection_experiments.evaluation.detection import parse_log_file


def attack_type_to_name(attack_type: Optional[str]) -> str:
    """
    Convert an attack type to a standardized name.

    Args:
        attack_type (Optional[str]): The attack type to convert.

    Returns:
        str: The standardized name of the attack type.

    Raises:
        ValueError: If the attack type is not recognized.
    """
    if attack_type is None:
        return "None"
    if attack_type not in list(Attackers.keys()):
        raise ValueError
    return attack_type


def initialize_empty_data_structure() -> dict:
    """
    Initialize an empty nested dictionary structure for storing experiment results.

    Returns:
        dict: An empty nested dictionary organized by experiment parameters.
    """
    extracted_data = {}
    for num_widths in [2, 4, 8]:
        extracted_data[num_widths] = {}
        for time_of_day in TimeOfDay:
            time_of_day = time_of_day.value
            extracted_data[num_widths][time_of_day] = {}
            for road_type in RoadType:
                road_type = road_type.value
                extracted_data[num_widths][time_of_day][road_type] = {}
                for attack_type in ["FullFrameInjection", "None"]:
                    extracted_data[num_widths][time_of_day][road_type][attack_type] = []
    return extracted_data


def access_nested_dict(nested_dict: dict, key: Tuple[Any, ...]) -> Any:
    """
    Access a nested dictionary using a tuple of keys.

    Args:
        nested_dict (dict): The nested dictionary to access.
        key (Tuple[Any, ...]): A tuple of keys to navigate the nested structure.

    Returns:
        Any: The value at the specified nested location.
    """
    current_level = nested_dict
    for k in key:
        current_level = current_level[k]
    return current_level


def extract_metadata_key_of_config_path(config_path: Path) -> Tuple:
    """
    Extract metadata keys from a configuration file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Tuple: A tuple containing (num_widths, time_of_day, road_type, attack_type).
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    num_widths = config["car"]["variation"]["num_widths"]
    time_of_day = config["experiment"]["time_of_day"]
    road_type = config["experiment"]["road_type"]
    attack_type = attack_type_to_name(config["attacker"]["attack_type"])
    return num_widths, time_of_day, road_type, attack_type


def extact_data_of_logs(
    base_dir: Path, extraction_function: Callable[[Path], Any]
) -> dict:
    """
    Extract data from log files using a specified extraction function.

    Args:
        base_dir (Path): Base directory containing experiment logs.
        extraction_function (Callable[[Path], Any]): Function to extract data from each log file.

    Returns:
        dict: A nested dictionary containing extracted data organized by experiment parameters.
    """
    extracted_data = initialize_empty_data_structure()
    for log_path in tqdm(list(base_dir.glob("*/log*"))):
        id = log_path.stem[4:]
        config_path = log_path.parent / f"config_{id}.yaml"
        key = extract_metadata_key_of_config_path(config_path)
        values = access_nested_dict(extracted_data, key)
        values.append(extraction_function(log_path))
    return extracted_data


def calc_number_of_experiments_success_rate_above_th(
    data: list, th: float
) -> Tuple[str, int]:
    """
    Calculate the number of experiments with a success rate above a given threshold.

    Args:
        data (list): List of success rates.
        th (float): Threshold for success rate.

    Returns:
        Tuple[str, int]: A tuple containing a description string and the count of experiments above the threshold.
    """
    return f"Sucess Rate Above {th}", len(list(filter(lambda x: x >= th, data)))


def get_completed_pcap_frames_ids(log_path: Path) -> List[int]:
    """
    Get the IDs of completed PCAP frames from a given log file.

    Args:
        log_path (Path): Path to the log file.

    Returns:
        List[int]: A list of completed frame IDs, or None if the directory doesn't exist.
    """
    completed_frames_directory = log_path.parent / "pcap_completed_frames"
    if not completed_frames_directory.exists():
        return None
    block_ids = [
        int(frame_path.stem.split("BlockID_")[1])
        for frame_path in completed_frames_directory.glob("*")
    ]
    block_ids = sorted(block_ids)
    return block_ids


def get_normal_completed_pcap_frames_ids(log_path: Path) -> List[int]:
    """
    Get the IDs of completed PCAP frames during normal operation (non-attack periods).

    Args:
        log_path (Path): Path to the log file.

    Returns:
        List[int]: A list of completed frame IDs during normal operation, or None if the directory doesn't exist.
    """
    completed_frames_directory = log_path.parent / "pcap_completed_frames"
    if not completed_frames_directory.exists():
        return None
    frames_df = parse_log_file(file_path=log_path, during_attack=False)
    frames_id = frames_df["frame_id"].to_list()

    block_ids = []
    for frame_path in completed_frames_directory.glob("*"):
        frame_id = int(frame_path.stem.split("_")[1])
        block_id = int(frame_path.stem.split("_")[3])
        if frame_id in frames_id:
            block_ids.append(block_id)
    block_ids = sorted(block_ids)
    return block_ids


def get_number_of_completed_frames(data: list) -> Tuple[str, int]:
    """
    Get the total number of completed frames across all experiments.

    Args:
        data (list): List of lists containing frame IDs for each experiment.

    Returns:
        Tuple[str, int]: A tuple containing a description string and the total number of completed frames.
    """
    return "Number of Completed Frames", sum(
        map(len, filter(lambda x: x is not None, data))
    )


def get_number_of_consecutive_frames(data: list) -> Tuple[str, int]:
    """
    Get the total number of consecutive frames across all experiments.

    Args:
        data (list): List of lists containing frame IDs for each experiment.

    Returns:
        Tuple[str, int]: A tuple containing a description string and the total number of consecutive frames.
    """

    def number_of_consecutives(lst: list) -> int:
        if data is None:
            return 0
        count = 0
        for i in range(1, len(lst)):
            count += lst[i] == (lst[i - 1] + 1)
        return count

    return "Number of Consecutive Frames", sum(map(number_of_consecutives, data))


if __name__ == "__main__":
    base_dir = Path()
    extracted_sucess_rate = extact_data_of_logs(
        base_dir=base_dir, extraction_function=evaluate_success_recording_rate
    )

    extracte_pcap_ids = extact_data_of_logs(
        base_dir=base_dir, extraction_function=get_normal_completed_pcap_frames_ids
    )

    success_rate_aggregation_functions = [
        partial(calc_number_of_experiments_success_rate_above_th, th=0),
        partial(calc_number_of_experiments_success_rate_above_th, th=0.5),
        partial(calc_number_of_experiments_success_rate_above_th, th=0.9),
        partial(calc_number_of_experiments_success_rate_above_th, th=0.99),
    ]

    pcap_frame_ids_aggregation_functions = [
        get_number_of_completed_frames,
        get_number_of_consecutive_frames,
    ]

    with open((base_dir / "summary.txt").as_posix(), "w") as f:
        for num_widths in [2, 4, 8]:
            for time_of_day in TimeOfDay:
                time_of_day = time_of_day.value
                for road_type in RoadType:
                    road_type = road_type.value
                    for attack_type in ["FullFrameInjection", "None"]:
                        key = num_widths, time_of_day, road_type, attack_type
                        f.write(f"{key}:\n")

                        data = access_nested_dict(extracted_sucess_rate, key)
                        for func in success_rate_aggregation_functions:
                            name, value = func(data)
                            f.write(f"\t{name}: {value}\n")

                        data = access_nested_dict(extracte_pcap_ids, key)
                        for func in pcap_frame_ids_aggregation_functions:
                            name, value = func(data)
                            f.write(f"\t{name}: {value}\n")
                        f.write(f"\n")
