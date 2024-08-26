"""
calibrate_using_dataset.py - Calibration Script for Stop Sign Detector

This module calibrates a stop sign detector using a dataset of annotated images.
It processes both positive (containing stop signs) and negative (without stop signs) 
images to generate a calibration dataset.

Key Components:
- DatasetImage: Class to handle image loading and resizing
- process_positive_annotation: Function to process images with stop signs
- process_negative_annotation: Function to process images without stop signs
- get_all_annotations: Function to load all image annotations
- split_annotations: Function to separate annotations into target and non-target

Dependencies:
- opencv-python (cv2): For image processing
- pandas: For data manipulation and CSV output
- tqdm: For progress bars
- multiprocessing: For parallel processing of images

Usage:
Run this script directly to generate a calibration dataset. The resulting CSV file
will contain confidence scores for both positive and negative samples.

Note: Ensure that the dataset directory structure and paths are correctly set up
before running the script.
"""

from typing import List, Tuple, Optional
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import json
import random
import cv2
import multiprocessing
from dataclasses import dataclass


from passive_detectors_evaluation.bootstrapper import DST_SHAPE
from sign_detectors import get_detector
from utils.datasets import get_largest_bounding_box
from utils.detection_utils import Rectangle, calculate_iou


dataset_directory = Path("../datasets/mtsd_v2_fully_annotated")
images_directory = dataset_directory / "images"
annotations_directory = dataset_directory / "annotations"
stop_sign_object = "regulatory--stop--g1"


detector = get_detector("MobileNet")
detector.confidence_th = 0
detector.nms_th = 0.01


iou_th = 0.3


def load_annotation_from_path(annotation_path: Path) -> dict:
    """
    Load a single annotation file from the given path.

    Args:
        annotation_path (Path): Path to the annotation JSON file.

    Returns:
        dict: Loaded annotation data with added 'image_key'.
    """
    with open(annotation_path.as_posix(), "r") as fid:
        anno = json.load(fid)
    anno["image_key"] = annotation_path.stem
    return anno


def get_all_annotations(max_files: Optional[int] = None) -> List[dict]:
    """
    Load all annotation files from the annotations directory.

    Args:
        max_files (Optional[int]): Maximum number of files to load. If None, load all.

    Returns:
        List[dict]: List of loaded annotation dictionaries.
    """
    all_files = list(annotations_directory.glob("*.json"))
    if max_files is not None:
        random.shuffle(all_files)
        all_files = all_files[:max_files]

    return [load_annotation_from_path(json_path) for json_path in tqdm(all_files)]


def split_annotations(
    annotations: List[dict], target_object: str
) -> Tuple[List[dict], List[dict]]:
    """
    Split annotations into target (containing the specified object) and non-target.

    Args:
        annotations (List[dict]): List of all annotation dictionaries.
        target_object (str): The target object label to filter for.

    Returns:
        Tuple[List[dict], List[dict]]: Lists of target and non-target annotations.
    """
    # filter only annotation of the target object
    target_annotations_files = []
    non_target_annotations = []
    for annotation in tqdm(annotations):
        if not (images_directory / f"{annotation['image_key']}.jpg").exists():
            continue
        if any(
            [
                labelled_object["label"] == target_object
                for labelled_object in annotation["objects"]
            ]
        ):
            target_annotations_files.append(annotation)
        else:
            non_target_annotations.append(annotation)

    # set only taret objects if few in same image
    target_annotations = []
    for annotation in tqdm(target_annotations_files):
        new_annotation = annotation.copy()
        new_annotation["objects"] = []
        for labelled_object in annotation["objects"]:
            if labelled_object["label"] == target_object:
                new_annotation["objects"].append(labelled_object)

        target_annotations.append(new_annotation)
    return target_annotations, non_target_annotations


@dataclass
class ImageShape:
    """Dataclass to represent image dimensions."""

    width: int
    height: int


class DatasetImage:
    """Class to handle dataset image loading and resizing."""

    def __init__(self, img_key: str) -> None:
        """
        Initialize DatasetImage with the given image key.

        Args:
            img_key (str): The key (filename without extension) of the image.
        """
        # read
        img_path = images_directory / f"{img_key}.jpg"
        img_bgr = cv2.imread(img_path.as_posix())
        height, width, _ = img_bgr.shape
        self._original_shape = ImageShape(width=width, height=height)
        # resize
        self.img = cv2.resize(img_bgr, DST_SHAPE)

    @property
    def shape(self) -> ImageShape:
        return ImageShape(width=self.img.shape[1], height=self.img.shape[0])

    @property
    def original_shape(self) -> ImageShape:
        return self._original_shape

    @property
    def resize_width_factor(self) -> float:
        return self.shape.width / self.original_shape.width

    @property
    def resize_height_factor(self) -> float:
        return self.shape.height / self.original_shape.height


def process_positive_annotation(annotation: dict) -> float:
    """
    Process a positive annotation (image containing a stop sign).

    Args:
        annotation (dict): Annotation dictionary for the image.

    Returns:
        float: Confidence score of the detection.
    """
    image_key = annotation["image_key"]
    img = DatasetImage(image_key)
    gt_bounding_box = get_largest_bounding_box(
        annotation=annotation, target_object=stop_sign_object
    )
    gt_bounding_box.resize(
        height_factor=img.resize_height_factor, width_factor=img.resize_width_factor
    )
    detections = detector.detect(img.img)
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

    if len(detections) > 0:
        pred_detection = detections[0]
        pred_bounding_box = Rectangle(
            detections[0].get_upper_left_corner(),
            detections[0].get_lower_right_corner(),
        )
        iou = calculate_iou(gt_bounding_box, pred_bounding_box)
        if iou > iou_th:
            pred_score = pred_detection.confidence
        else:
            pred_score = 0
    else:
        pred_score = 0

    return pred_score


def process_negative_annotation(annotation: dict) -> float:
    """
    Process a negative annotation (image not containing a stop sign).

    Args:
        annotation (dict): Annotation dictionary for the image.

    Returns:
        float: Confidence score of the highest detection (if any).
    """
    image_key = annotation["image_key"]
    img = DatasetImage(image_key)
    detections = detector.detect(img.img)
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    if len(detections) > 0:
        pred_detection = detections[0]
        pred_score = pred_detection.confidence
    else:
        pred_score = 0

    return pred_score


if __name__ == "__main__":
    annotations = get_all_annotations(max_files=None)
    target_annotations, other_annotations = split_annotations(
        annotations, target_object=stop_sign_object
    )

    num_non_target = 5000
    sampled_other_annotations = random.sample(other_annotations, num_non_target)
    scores = []
    labels = []

    with multiprocessing.Pool(4) as pool:
        for res in tqdm(
            pool.imap(process_positive_annotation, target_annotations),
            total=len(target_annotations),
        ):
            scores.append(res)
            labels.append(True)

    with multiprocessing.Pool(4) as pool:
        for res in tqdm(
            pool.imap(process_negative_annotation, sampled_other_annotations),
            total=len(sampled_other_annotations),
        ):
            scores.append(res)
            labels.append(False)

    df = pd.DataFrame({"scores": scores, "labels": labels})
    df.to_csv("../OUTPUT/sign_detector_calibration_overlapped.csv", index=False)
