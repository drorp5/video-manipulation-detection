"""
cli.py - Command Line Interface for Active Detection Experiments

This module provides a command-line interface for running active detection experiments.
It includes functionality to parse command-line arguments, run experiment iterations,
and play sounds based on the experiment's success or failure.

Key Components:
- run_experiment_iterarion_wrapper: Runs a single experiment iteration
- parse_args: Parses command-line arguments
- main: Main entry point for the CLI

Usage:
python active_detction_experiments/cli.py --num_widths <number_of_widths> [--attack_type <attack_type>]

Dependencies:
- playsound: For playing success/failure sounds
- active_detection_experiments.run_experiment: For running the actual experiment
"""

from typing import Optional
import argparse
import sys
import yaml
from pathlib import Path
from playsound import playsound
from active_detection_experiments.run_experiment import run_experiment


SUCCESS_SOUND_PATH = Path(r"../INPUT/success.mp3")
FAILURE_SOUND_PATH = Path(r"../INPUT/failure.mp3")
BASE_CONFIG_PATH = Path(
    r"../active_detection_experiments/day_urban_experiment_config.yaml"
)


def run_experiment_iterarion_wrapper(
    num_widths: int, attack_type: Optional[str] = None, success_rate_th: float = 0.9
) -> bool:
    """
    Run a single iteration of the experiment with the given parameters.

    Args:
        num_widths (int): Number of possible image widths.
        attack_type (Optional[str]): Type of attack to simulate. Defaults to None.
        success_rate_th (float): Threshold for considering the experiment successful. Defaults to 0.9.

    Returns:
        bool: True if the experiment was successful, False otherwise.
    """
    with open(BASE_CONFIG_PATH.as_posix(), "r") as f:
        experiment_config = yaml.safe_load(stream=f)
    experiment_config["car"]["variation"]["num_widths"] = num_widths
    experiment_config["attacker"]["attack_type"] = attack_type

    experiment = run_experiment(experiment_config)
    success_rate = experiment.evaluate_success_rate()
    print(f"Success Rate = {success_rate}")
    success_flag = success_rate >= success_rate_th

    if success_flag:
        playsound(SUCCESS_SOUND_PATH.as_posix())
    else:
        playsound(FAILURE_SOUND_PATH.as_posix())
    return success_flag


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_widths", help="number of possible image widths", type=int
    )
    parser.add_argument(
        "--attack_type", help="type of injection attack", type=str, default=None
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        return run_experiment_iterarion_wrapper(
            num_widths=args.num_widths,
            attack_type=args.attack_type,
            success_rate_th=0.9,
        )
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    suceess_flag = main()
    if suceess_flag:
        sys.exit(0)
    else:
        sys.exit(1)
