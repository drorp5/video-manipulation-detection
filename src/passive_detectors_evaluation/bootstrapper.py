import cv2
from typing import List
from pathlib import Path
from injectors.full_frame_injector import FullFrameInjector
from injectors.injector import Injector
from injectors.stop_sign_injector import SignPatchInjector
from injectors.stripe_injector import StripeInjector
from injectors.rectangular_patch_injector import (
    RectangularPatchInjector,
)
from passive_detectors_evaluation.datasets import (
    EvaluationDataset,
    FramesDirectoryDataset,
    VideoDataset,
    VideosDirectoryDataset,
)

from passive_detectors.image_processing import (
    OpticalFlowDetector,
    HueSaturationHistogramDetector,
    ImageProcessingDetector,
)
from gige.gige_constants import MAX_HEIGHT, MAX_WIDTH

DST_SHAPE = (MAX_WIDTH, MAX_HEIGHT)

VIDEOS_PATH = Path(r"Datasets/BDD100K/bdd100k_videos_test_00/bdd100k/videos/test")


def get_stop_sign_injector(injector_type: str) -> Injector:
    """
    Create and return a stop sign injector based on the specified type.

    Args:
        injector_type (str): Type of injector ('full_frame', 'stripe', or 'patch').

    Returns:
        Injector: An injector object of the specified type.

    Raises:
        ValueError: If an invalid injector_type is provided.
    """
    if injector_type == "full_frame":
        stop_sign_road = cv2.imread(r"INPUT/stop_sign_road_2.jpg")
        stop_sign_road = cv2.cvtColor(stop_sign_road, cv2.COLOR_BGR2RGB)
        return FullFrameInjector(fake_img=stop_sign_road, dst_shape=DST_SHAPE)
    elif injector_type == "stripe":
        stop_sign_road = cv2.imread(r"INPUT/stop_sign_road_2.jpg")
        stop_sign_road = cv2.resize(stop_sign_road, DST_SHAPE)
        stop_sign_road = cv2.cvtColor(stop_sign_road, cv2.COLOR_BGR2RGB)
        return StripeInjector(fake_img=stop_sign_road, first_row=354, last_row=489)
    elif injector_type == "patch":
        stop_sign = cv2.imread(r"INPUT/stop_sign_road_2_resized_cropped.jpg")
        sign_img = cv2.cvtColor(stop_sign, cv2.COLOR_BGR2RGB)
        side_length = 108
        first_row = 4
        last_row = 138
        return SignPatchInjector(
            sign_img=sign_img,
            side_length=side_length,
            first_row=first_row,
            last_row=last_row,
        )
    else:
        raise ValueError


def get_red_light_injector(injector_type: str) -> Injector:
    """
    Create and return a red light injector based on the specified type.

    Args:
        injector_type (str): Type of injector ('full_frame', 'stripe', or 'patch').

    Returns:
        Injector: An injector object of the specified type.

    Raises:
        ValueError: If an invalid injector_type is provided.
    """
    red_light_path = r"INPUT/red_light.jpg"
    red_light_img = cv2.imread(red_light_path)
    red_light_img = cv2.cvtColor(red_light_img, cv2.COLOR_BGR2RGB)

    if injector_type == "full_frame":
        return FullFrameInjector(fake_img=red_light_img, dst_shape=DST_SHAPE)
    elif injector_type == "stripe":
        return StripeInjector(fake_img=red_light_img, first_row=133, last_row=197)
    elif injector_type == "patch":
        return RectangularPatchInjector(
            patch_img=red_light_img,
            first_row=133,
            last_row=197,
            first_col=810,
            last_col=875,
        )
    else:
        raise ValueError


def get_image_processing_detectors() -> List[ImageProcessingDetector]:
    """
    Create and return a list of image processing detectors.

    Returns:
        List[ImageProcessingDetector]: A list containing histogram and optical flow detectors.
    """
    hist_detector = HueSaturationHistogramDetector(0)
    optical_flow_detector = OpticalFlowDetector(0)
    return [hist_detector, optical_flow_detector]


def get_dataset(data_source: str) -> EvaluationDataset:
    """
    Create and return a dataset based on the specified data source.

    Args:
        data_source (str): Source of the dataset ('experiment' or 'BDD').

    Returns:
        EvaluationDataset: A dataset object of the specified type.

    Raises:
        ValueError: If an invalid data_source is provided.
    """
    if data_source == "experiment":
        frames_dir = Path(r"OUTPUT/drive_around_uni_frames/")
        dataset_name = "drive_around_uni"
        min_frame_id = 1800
        return FramesDirectoryDataset(
            frames_dir=frames_dir, name=dataset_name, min_frame_id=min_frame_id
        )
    elif data_source == "BDD":
        num_frames = 100
        flip = True
        return VideosDirectoryDataset(
            videos_dir=VIDEOS_PATH,
            name="BDD",
            num_frames=num_frames,
            flip=flip,
            dst_shape=DST_SHAPE,
        )
    else:
        raise ValueError
