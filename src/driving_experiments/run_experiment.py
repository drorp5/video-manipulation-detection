import json
import logging
import threading
import yaml
from pathlib import Path
import numpy as np
import random

from car.changing_shape_defense_car import ShapeVaryingLogicCar
from active_manipulation_detectors.side_channel.data_generator import (
    RandomBitsGeneratorRC4,
)
from active_manipulation_detectors.side_channel.validation import (
    DataValidatorKSymbolsDelayed,
)
from attacker import GigEAttackerStripeInjection
from driving_experiments.experiment import Experiment
from active_manipulation_detectors.evaluation.mtsd_evaluation import (
    get_largest_bounding_box,
    resize_bounding_box,
)
from active_manipulation_detectors.evaluation.metadata import DATASET_TO_TARGET_OBJECT


def run_experiment_using_config_path(config_path: Path) -> None:
    with open(config_path.as_posix(), "r") as f:
        experiment_config = yaml.safe_load(stream=f)
    run_experiment(experiment_config)


def fill_attacker_config(config: dict) -> None:
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


def run_experiment(experiment_config: dict) -> None:
    # logger
    logger = logging.getLogger(f"experiment_logger")
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

    data_validator = DataValidatorKSymbolsDelayed(
        bits_in_symbol=num_bits_per_iteration,
        symbols_for_detection=car_config["validator"]["num_symbols"],
        max_delay=car_config["validator"]["max_delay"],
    )
    
    streaming_stopped_event = threading.Event()
    car_logic = ShapeVaryingLogicCar(
        config=experiment_config["car"],
        random_bits_generator=random_bits_generator,
        data_validator=data_validator,
        logger=logger,
        external_event=streaming_stopped_event
    )

    # attacker
    fill_attacker_config(experiment_config)
    attacker = GigEAttackerStripeInjection(experiment_config["attacker"], logger=logger)

    # set experiment
    experiment = Experiment(
        config=experiment_config,
        logger=logger,
        car=car_logic,
        attacker=attacker,
    )

    experiment.run()


if __name__ == "__main__":
    # experiment configuration
    experiment_configuration_path = r"driving_experiments/experiment_config.yaml"
    run_experiment_using_config_path(config_path=experiment_configuration_path)
