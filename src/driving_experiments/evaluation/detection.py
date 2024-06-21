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


from active_manipulation_detectors.validation_status import ValidationStatus
from driving_experiments.tags import TimeOfDay, RoadType


injection_started_log_line = "Full Frame Injection Started"
injection_ended_log_line = "Full Frame Injection Ended"


def parse_log_file(file_path: Path, during_attack: bool) -> pd.DataFrame:
    with open(file_path.as_posix(), "r") as file:
        log_lines = file.readlines()

    log_pattern = r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<level>\w+) - (?P<message>.*)$"
    parsed_data = []

    currently_during_injection = False
    for line in log_lines:
        if injection_started_log_line in line:
            currently_during_injection = True
        elif injection_ended_log_line in line:
            currently_during_injection = False

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

                if during_attack and currently_during_injection:
                    parsed_data.append(frame_data)
                elif not during_attack and not currently_during_injection:
                    parsed_data.append(frame_data)

    return pd.DataFrame(parsed_data)


@dataclass
class DetectionEvaluationResult:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    incomplete: int = 0

    def __add__(self, other):
        if not isinstance(other, DetectionEvaluationResult):
            return NotImplemented
        return DetectionEvaluationResult(
            self.total + other.total,
            self.valid + other.valid,
            self.invalid + other.invalid,
            self.incomplete + other.incomplete,
        )

    def __iadd__(self, other):
        if not isinstance(other, DetectionEvaluationResult):
            return NotImplemented
        self.total += other.total
        self.valid += other.valid
        self.invalid += other.invalid
        self.incomplete += other.incomplete
        return self

    def __repr__(self):
        return f"DetectionEvaluationResult(total={self.total}, valid={self.valid}, invalid={self.invalid}, incomplete={self.incomplete})"


class DetectionEvaluator:
    def __init__(
        self,
        base_dir: Path,
        num_widths: int,
        time_of_day: TimeOfDay,
        road_type: RoadType,
    ) -> None:
        self.base_dir = base_dir
        self.num_widths = num_widths
        self.time_of_day = time_of_day
        self.road_type = road_type

    def evaluate_log(
        self, log_path: Path, during_attack: bool
    ) -> DetectionEvaluationResult:
        log_df = parse_log_file(log_path, during_attack)
        if len(log_df) == 0:
            return DetectionEvaluationResult()
        validation_results_temp = log_df["validation_result"].value_counts().to_dict()
        all_possible_values = ValidationStatus._member_names_
        validation_results = {
            k.lower(): validation_results_temp.get(k, 0) for k in all_possible_values
        }
        validation_results["total"] = len(log_df)
        return DetectionEvaluationResult(**validation_results)

    def _is_experiment_for_evaluation(self, experiment_dir: Path) -> bool:
        config_path = list(experiment_dir.glob("config*"))[0]
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        if not config["experiment"]["time_of_day"] == self.time_of_day.value:
            return False
        if not config["experiment"]["road_type"] == self.road_type.value:
            return False
        if not config["car"]["variation"]["num_widths"] == self.num_widths:
            return False
        return True

    def evalutate_dir(
        self,
    ) -> Tuple[DetectionEvaluationResult, DetectionEvaluationResult]:
        during_attack_logs_res = []
        normal_logs_res = []
        logs_path = list(self.base_dir.glob("*/log*"))
        for log_path in tqdm(logs_path):
            if not self._is_experiment_for_evaluation(log_path.parent):
                continue
            normal_logs_res.append(self.evaluate_log(log_path, during_attack=False))
            during_attack_logs_res.append(
                self.evaluate_log(log_path, during_attack=True)
            )
        initial_value = DetectionEvaluationResult()
        combined_normal_res = sum(normal_logs_res, initial_value)
        combined_attacked_res = sum(during_attack_logs_res, initial_value)
        return combined_normal_res, combined_attacked_res


if __name__ == "__main__":
    base_dir = Path(r"D:\Thesis\video-manipulation-detection\driving_experiments")

    with open(base_dir / "evaluation.txt", "w") as f:

        for num_widths in [2, 4, 8]:
            total_normal = DetectionEvaluationResult()
            total_attacked = DetectionEvaluationResult()
            for time_of_day in TimeOfDay:
                for road_type in RoadType:
                    evaluator = DetectionEvaluator(
                        base_dir,
                        num_widths=num_widths,
                        time_of_day=time_of_day,
                        road_type=road_type,
                    )
                    normal_res, attack_res = evaluator.evalutate_dir()
                    total_normal += normal_res
                    total_attacked += attack_res
                    f.write(
                        f"NumWidths={num_widths}\nTimeOfDay: {time_of_day.value}\nRoadType: {road_type.value}\nNormal: {normal_res}\nAttack: {attack_res}\n"
                    )
            f.write(
                f"NumWidths={num_widths}\nNormal: {total_normal}\nAttack: {total_attacked}\n"
            )
            f.write(
                f"detection_probability = {total_attacked.invalid / total_attacked.total}\n"
            )
            f.write(f"expected_detection_probability = {(1-1/num_widths)**2}\n")
