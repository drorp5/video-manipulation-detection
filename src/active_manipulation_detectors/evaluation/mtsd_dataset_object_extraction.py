from pathlib import Path
import shutil
from tqdm import tqdm
from icecream import ic
from pprint import pprint
from typing import List, Dict, Tuple
import cv2
import os
import json
import shutil
import multiprocessing
from functools import partial

from detectors_evaluation.bootstrapper import DST_SHAPE
from sign_detectors import StopSignDetector, get_detector, draw_bounding_boxes
from utils.detection_utils import calculate_iou, Rectangle
from utils.image_processing import bgr_to_bayer_rg


dataset_directory = Path("../datasets/mtsd_v2_fully_annotated")
images_directory = dataset_directory / "images"
annotations_directory = dataset_directory / "annotations"


def load_annotation_from_key(image_key: str) -> dict:
    annotation_path = annotations_directory / f"{image_key}.json"
    return load_annotation_from_path(annotation_path)


def load_annotation_from_path(annotation_path: Path) -> dict:
    with open(annotation_path.as_posix(), "r") as fid:
        anno = json.load(fid)
    anno["image_key"] = annotation_path.stem
    return anno


def get_target_annotations(target_object: str) -> List[Dict]:
    annotations = [
        load_annotation_from_path(json_path)
        for json_path in tqdm(list(annotations_directory.glob("*.json")))
    ]

    # filter only annotation of the target object
    target_annotations_files = list(
        filter(
            lambda annotation: any(
                [
                    labelled_object["label"] == target_object
                    for labelled_object in annotation["objects"]
                ]
            ),
            annotations,
        )
    )

    # filter only annotations with existing images
    target_annotations_files = list(
        filter(
            lambda annotation: (
                images_directory / f"{annotation['image_key']}.jpg"
            ).exists(),
            target_annotations_files,
        )
    )

    # set only taret objects if few in same image
    target_annotations = []
    for annotation in tqdm(target_annotations_files):
        new_annotation = annotation.copy()
        new_annotation["objects"] = []
        for labelled_object in annotation["objects"]:
            if labelled_object["label"] == target_object:
                new_annotation["objects"].append(labelled_object)

        target_annotations.append(new_annotation)
    return target_annotations


def copy_images_to_directory(annotations: List[dict], dst_dir: Path) -> None:
    for anno in tqdm(annotations):
        source_file = images_directory / f'{anno["image_key"]}.jpg'
        destination_file = dst_dir / f'{anno["image_key"]}.jpg'
        shutil.copyfile(source_file, destination_file)


def resize_images_and_save_to_directory(annotations: List[dict], dst_dir: Path) -> None:
    for anno in tqdm(annotations):
        source_file = images_directory / f'{anno["image_key"]}.jpg'
        img_bgr = cv2.imread(source_file.as_posix())
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, DST_SHAPE)
        destination_file = dst_dir / f'{anno["image_key"]}.jpg'
        cv2.imwrite(
            destination_file.as_posix(), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        )


def run_sign_detection(
    annotation: dict,
    dst_shape: Tuple[int, int] = DST_SHAPE,
    iou_th: float = 0.5,
) -> None:
    """run sign detector on specific frame, and if detection matches annotation bounding box, saves the reshaped image in directory"""
    source_file = images_directory / f'{annotation["image_key"]}.jpg'
    img_bgr = cv2.imread(source_file.as_posix())

    # reisze image
    original_height, original_width, _ = img_bgr.shape
    img_bgr = cv2.resize(img_bgr, dst_shape)
    new_width, new_height = dst_shape

    # convert to bayer and then to RGB to simulate real frame
    bayer_img = bgr_to_bayer_rg(img_bgr)
    img_rgb = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2RGB)

    # resize ground truth bounding boxes
    width_resizing_factor = new_width / original_width
    height_resizing_factor = new_height / original_height
    gt_detections = []
    for object in annotation["objects"]:
        gt_bounding_box_annotation = object["bbox"]
        gt_bounding_box = Rectangle(
            (gt_bounding_box_annotation["xmin"], gt_bounding_box_annotation["ymin"]),
            (gt_bounding_box_annotation["xmax"], gt_bounding_box_annotation["ymax"]),
        )
        gt_bounding_box.resize(width_resizing_factor, height_resizing_factor)
        gt_detections.append(gt_bounding_box)

    # check for matching detections
    detections = detector.detect(img_rgb)
    matched = False
    for detection in detections:
        x, y, w, h = detection
        pred_detection = Rectangle((x, y), (x + w, y + h))
        for gt_detection in gt_detections:
            iou = calculate_iou(pred_detection, gt_detection)
            if iou >= iou_th:
                matched = True
                break
        if matched:
            break

    if matched:
        # copy image
        destination_dir = dataset_directory / f"{detector.name}_detections"
        if not destination_dir.exists():
            destination_dir.mkdir(parents=True)
        destination_file = destination_dir / f"{annotation['image_key']}.jpg"
        cv2.imwrite(
            destination_file.as_posix(),
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        )

        # copy image and markk bounding boxes
        destination_dir = dataset_directory / f"{detector.name}_detections_marked"
        if not destination_dir.exists():
            destination_dir.mkdir(parents=True)

        img_with_detections = draw_bounding_boxes(img_rgb, detections)
        destination_file = destination_dir / f"{annotation['image_key']}.jpg"
        cv2.imwrite(
            destination_file.as_posix(),
            cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR),
        )


detector = get_detector("MobileNet")

if __name__ == "__main__":
    target_annotations = get_target_annotations(target_object="regulatory--stop--g1")
    print(f"Total {len(target_annotations)} annotations of target class")

    # dst_dir = dataset_directory / "stop_sign_images"
    # copy_images_to_directory(target_annotations, dst_dir)

    # dst_dir = dataset_directory / "stop_sign_images_resized"
    # resize_images_and_save_to_directory(target_annotations, dst_dir)

    with multiprocessing.Pool(4) as p:
        for res in tqdm(
            p.imap_unordered(run_sign_detection, target_annotations),
            total=len(target_annotations),
        ):
            pass
