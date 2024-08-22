from datetime import datetime
import json
import logging
import threading
from typing import Optional
import yaml
from pathlib import Path
import numpy as np
import random
import uuid

import vimba

from car.changing_shape_defense_car import ShapeVaryingLogicCar
from active_manipulation_detectors.side_channel.data_generator import (
    RandomBitsGeneratorRC4,
)
from active_manipulation_detectors.side_channel.validation import (
    DataValidatorKSymbolsDelayed,
)
from active_manipulation_detectors.side_channel.validation import (
    DataValidatorKSymbolsDelayedChanged,
)
from attacker import GigEAttackerStripeInjection, GigEAttackerFrameInjection, Attackers
from driving_experiments.experiment import Experiment
from active_manipulation_detectors.evaluation.mtsd_evaluation import (
    get_largest_bounding_box,
    resize_bounding_box,
)
from active_manipulation_detectors.evaluation.metadata import DATASET_TO_TARGET_OBJECT
from gige.gige_constants import MAX_HEIGHT, MAX_WIDTH
from recorders import VideoReocrder, FramesRecorder


def run_experiment_using_config_path(config_path: Path) -> None:
    with open(config_path.as_posix(), "r") as f:
        experiment_config = yaml.safe_load(stream=f)
    run_experiment(experiment_config)


def fill_attacker_config(config: dict) -> None:
    config["attacker"]["timing"]["fps"] = config["car"]["camera"]["fps"]
    if config["attacker"]["timing"]["pre_attack_duration_in_seconds"] is None:
        config["attacker"]["timing"]["pre_attack_duration_in_seconds"] = 0
    if config["attacker"]["timing"]["attack_duration_in_seconds"] is None:
        config["attacker"]["timing"]["attack_duration_in_seconds"] = (
            config["experiment"]["duration"]
            - config["attacker"]["timing"]["pre_attack_duration_in_seconds"]
        )
    if config["attacker"]["injection"]["fake_path"] is None:
        # draw randomly from dir
        fake_dir = Path(config["attacker"]["injection"]["dataset"]["images_dir"])
        all_fake_path = list(fake_dir.glob("*.jpg"))
        fake_path = random.choice(all_fake_path)
        config["attacker"]["injection"]["fake_path"] = fake_path.as_posix()
        if config["attacker"]["attack_type"] != "FullFrameInjection":
            # read annotation
            annotation_path = (
                Path(config["attacker"]["injection"]["dataset"]["annotations_dir"])
                / f"{fake_path.stem}.json"
            )
            with open(annotation_path.as_posix(), "r") as fid:
                annotation = json.load(fid)
            # get largest bounding box from annotation
            gt_bounding_box = get_largest_bounding_box(
                annotation=annotation,
                target_object=DATASET_TO_TARGET_OBJECT[
                    config["attacker"]["injection"]["dataset"]["name"]
                ],
            )
            # resize bounding box to gvsp shape
            new_shape = (
                config["attacker"]["gige"]["gvsp"]["width"],
                config["attacker"]["gige"]["gvsp"]["height"],
            )
            resize_bounding_box(
                gt_bounding_box,
                old_shape=(annotation["width"], annotation["height"]),
                new_shape=new_shape,
            )
            # set injection params
            config["attacker"]["injection"]["first_row"] = gt_bounding_box.ymin
            config["attacker"]["injection"]["num_rows"] = (
                gt_bounding_box.ymax - gt_bounding_box.ymin
            )


def run_experiment(experiment_config: dict) -> Experiment:
    # generte experiment id
    experiment_id = str(uuid.uuid4())
    experiment_config["experiment"]["id"] = experiment_id

    # set start time
    now = datetime.now()
    start_time_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_config["experiment"]["start_time"] = start_time_string

    # set output directory
    base_results_dir = (
        Path(experiment_config["experiment"]["results_directory"])
        / f"{start_time_string}_{experiment_id}"
    )
    base_results_dir.mkdir(parents=True, exist_ok=True)
    experiment_config["experiment"][
        "results_directory"
    ] = base_results_dir.absolute().as_posix()

    # logger
    logger = logging.getLogger(experiment_id)
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    

    # car
    car_config = experiment_config["car"]
    key = car_config["variation"]["key"].encode("utf-8")
    num_symbols = car_config["variation"]["num_widths"]
    num_bits_per_iteration = int(np.ceil(np.log2(num_symbols)))
    random_bits_generator = RandomBitsGeneratorRC4(
        key=key, num_bits_per_iteration=num_bits_per_iteration
    )

    data_validator = DataValidatorKSymbolsDelayedChanged(
        symbols_for_detection=car_config["validator"]["num_symbols"],
        max_delay=car_config["validator"]["max_delay"],
        data_holder_type="list",
    )
    camera_started_event = threading.Event()
    camera_stopped_event = threading.Event()

    if car_config["actions"]["recorder"] == "video":
        downfactor = 4
        video_path = (
            Path(experiment_config["experiment"]["results_directory"])
            / f"{experiment_config['experiment']['id']}.mp4"
        )
        car_config["actions"]["video_path"] = video_path.absolute().as_posix()
        video_shape = (MAX_WIDTH // downfactor, MAX_HEIGHT // downfactor)
        recorder = VideoReocrder(
                video_path=video_path,
                fps=car_config["camera"]["fps"],
                video_shape=video_shape,
            )
    elif car_config["actions"]["recorder"] == "frames":
        frames_dir =  Path(experiment_config["experiment"]["results_directory"]) / "frames"
        recorder = FramesRecorder(frames_dir)
    elif car_config["actions"]["recorder"]  is not None :
        raise ValueError("recorder type not valid")
    else:
        recorder = None


    car_logic = ShapeVaryingLogicCar(
        config=experiment_config["car"],
        random_bits_generator=random_bits_generator,
        data_validator=data_validator,
        logger=logger,
        camera_started_event=camera_started_event,
        camera_stopped_event=camera_stopped_event,
        recorder=recorder
    )

    # attacker
    fill_attacker_config(experiment_config)
    attack_type = experiment_config["attacker"]["attack_type"]
    if attack_type is None:
        attacker = None
    else:
        attacker = Attackers[attack_type](
            experiment_config["attacker"],
            logger=logger,
            initialization_event=camera_started_event,
        )

    # set experiment
    experiment = Experiment(
        id=experiment_id,
        base_results_dir=base_results_dir,
        config=experiment_config,
        logger=logger,
        car=car_logic,
        attacker=attacker,
    )

    experiment.run()

    return experiment


if __name__ == "__main__":
    # experiment configuration
    experiment_configuration_path = r"driving_experiments/experiment_config.yaml"
    run_experiment_using_config_path(config_path=experiment_configuration_path)
