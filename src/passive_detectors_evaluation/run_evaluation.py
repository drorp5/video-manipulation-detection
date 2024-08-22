"""
This script runs an evaluation of passive video manipulation detection methods on different datasets.
It supports various injector types and can save examples of manipulated frames.
"""

from enum import Enum
from pathlib import Path
import cv2
from tqdm import tqdm
import multiprocessing
from functools import partial
import pandas as pd
from injectors import Injector, InjectorType
from passive_detectors_evaluation import (
    Evaluator,
    EvaluationDataset,
    Label,
    evaluate_pair,
)
import passive_detectors_evaluation.bootstrapper as bootstrapper


class ManipulationObject(Enum):
    STOP_SIGN = "stop_sign"
    RED_LIGHT = "red_light"


def run_evaluation(
    evaluator: Evaluator, dataset: EvaluationDataset, dst_dir_path: Path
):
    """
    Run the evaluation process on a given dataset using the specified evaluator.

    Args:
        evaluator (Evaluator): The evaluator object containing detectors and injector.
        dataset (EvaluationDataset): The dataset to evaluate on.
        dst_dir_path (Path): The directory to save results.

    Returns:
        None
    """
    res_by_detector = {
        detector.name: {"score": [], "label": []} for detector in evaluator.detectors
    }

    helper = partial(evaluate_pair, evaluator)
    with multiprocessing.Pool(6) as pool:
        for res in tqdm(pool.imap(helper, dataset), total=len(dataset)):
            for detector_res in res:
                res_by_detector[detector_res.detector]["label"].extend(
                    [Label.REAL.value, Label.FAKE.value]
                )
                res_by_detector[detector_res.detector]["score"].extend(
                    [detector_res.real, detector_res.fake]
                )

    # save results
    for detector in evaluator.detectors:
        df = pd.DataFrame(res_by_detector[detector.name])
        dst_path = (
            dst_dir_path
            / f"{dataset.name}_{evaluator.injector.name}_{detector.name}.csv"
        )
        df.to_csv(dst_path)


def save_example_frame(
    dataset: EvaluationDataset, injector: Injector, dst_path: Path
) -> None:
    """
    Save an example of a manipulated frame.

    Args:
        dataset (EvaluationDataset): The dataset to take the example from.
        injector (Injector): The injector to use for manipulation.
        dst_path (Path): The path to save the example frame.

    Returns:
        None
    """
    fake_frame = injector.inject(frame_1=dataset[0][0], frame_2=dataset[0][1])
    fake_frame_bgr = cv2.cvtColor(fake_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst_path.as_posix(), fake_frame_bgr)
    print(f"Example frame saved in {dst_path}")


def get_injector(injector_type: InjectorType, manipulation_object: ManipulationObject):
    """
    Get the appropriate injector based on the injector type and manipulation object.

    Args:
        injector_type (InjectorType): The type of injector to use.
        manipulation_object (ManipulationObject): The object to inject (stop sign or red light).

    Returns:
        Injector: The selected injector object.
    """
    if manipulation_object == ManipulationObject.STOP_SIGN:
        return bootstrapper.get_stop_sign_injector(injector_type.value)
    elif manipulation_object == ManipulationObject.RED_LIGHT:
        return bootstrapper.get_red_light_injector(injector_type.value)
    else:
        raise ValueError(f"Unknown manipulation object: {manipulation_object}")


def main():
    base_dir = Path("../OUTPUT")
    project_name = "video_manipulation_detection"
    save_example_only = False
    data_sources = ["BDD"]  # Can be expanded to include "experiment"
    injector_types = [InjectorType.FULL_FRAME, InjectorType.STRIPE, InjectorType.PATCH]
    manipulation_object = (
        ManipulationObject.RED_LIGHT
    )  # Can be changed to ManipulationObject.STOP_SIGN

    dst_dir = base_dir / project_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    detectors = bootstrapper.get_image_processing_detectors()

    for data_source in data_sources:
        for injector_type in injector_types:
            print(f"Running {injector_type.value} on {data_source}")

            dataset = bootstrapper.get_dataset(data_source)
            injector = get_injector(injector_type, manipulation_object)

            if save_example_only:
                dst_path = (
                    dst_dir
                    / f"fake_{data_source}_{injector.name}_{manipulation_object.value}.jpg"
                )
                save_example_frame(dataset, injector, dst_path)
            else:
                evaluator = Evaluator(detectors=detectors, injector=injector)
                run_evaluation(
                    evaluator=evaluator, dataset=dataset, dst_dir_path=dst_dir
                )


if __name__ == "__main__":
    main()
