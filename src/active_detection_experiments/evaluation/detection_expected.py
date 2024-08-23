from dataclasses import dataclass
from math import log
from pathlib import Path
from datetime import datetime
import re
import ast
from typing import Tuple
import pandas as pd
from sklearn import base
from tqdm import tqdm
import yaml


# from driving_experiments.tags import TimeOfDay, RoadType
from active_detection_experiments.evaluation.detection import parse_log_file
from active_detection_experiments.evaluation.experiments_summary import (
    extract_metadata_key_of_config_path,
)


if __name__ == "__main__":
    base_dir = Path(r"D:\Thesis\video-manipulation-detection\driving_experiments")

    counters = {
        2: {"total": 0, "full_width": 0, "total_delayed": 0, "successfull_attacks": 0},
        4: {"total": 0, "full_width": 0, "total_delayed": 0, "successfull_attacks": 0},
        8: {"total": 0, "full_width": 0, "total_delayed": 0, "successfull_attacks": 0},
    }

    for log_path in tqdm(list(base_dir.glob("*/log*"))):
        id = log_path.stem[4:]
        config_path = log_path.parent / f"config_{id}.yaml"
        num_widths, time_of_day, road_type, attack_type = (
            extract_metadata_key_of_config_path(config_path)
        )
        frames_df = parse_log_file(log_path, during_attack=True)
        if len(frames_df) > 0:
            counters[num_widths]["total"] += len(frames_df)
            counters[num_widths]["full_width"] += (
                frames_df["next_width"] == 1936
            ).sum()

            for i in range(2, len(frames_df)):
                counters[num_widths]["total_delayed"] += 1
                if (
                    frames_df["next_width"][i - 1] == 1936
                    or frames_df["next_width"][i - 2] == 1936
                ):
                    counters[num_widths]["successfull_attacks"] += 1

    for num_widths in [2, 4, 8]:
        full_width_probability = (
            counters[num_widths]["full_width"] / counters[num_widths]["total"]
        )
        print(f"{num_widths}:\n")
        print(
            f"\tFull Width Probabiity = {full_width_probability} ({counters[num_widths]['full_width']} of {counters[num_widths]['total']})\n"
        )
        print(f"\testimated_detection_probability = {(1-full_width_probability)**2}\n")

        ampriric_detection_probability = (
            1
            - counters[num_widths]["successfull_attacks"]
            / counters[num_widths]["total_delayed"]
        )
        print(
            f"\tampiric_detection_probability = {ampriric_detection_probability} ({counters[num_widths]['total_delayed'] - counters[num_widths]['successfull_attacks']} of {counters[num_widths]['total_delayed']})\n\n"
        )

    # true_0_distribution = {2: 0.57333, 4: 0.28, 8: 0.12666666}
