from typing import Optional
import argparse
import sys
import yaml
from pathlib import Path
from playsound import playsound

sys.path.append(".")
from active_detection_experiments.run_experiment import run_experiment


SUCCESS_SOUND_PATH = Path(r"C:\Users\user\Desktop\Dror\video-manipulation-detection\INPUT\success.mp3")
FAILURE_SOUND_PATH = Path(r"C:\Users\user\Desktop\Dror\video-manipulation-detection\INPUT\failure.mp3")
BASE_CONFIG_PATH = Path(r"C:\Users\user\Desktop\Dror\video-manipulation-detection\src\driving_experiments\day_urban_experiment_config.yaml")


def run_experiment_iterarion_wrapper(
    num_widths: int, attack_type: Optional[str] = None, success_rate_th: float = 0.9
) -> bool:
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
