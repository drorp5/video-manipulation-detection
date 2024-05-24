from pathlib import Path
import shutil
from tqdm import tqdm
from icecream import ic
from pprint import pprint
from typing import List, Dict
import cv2
import os
import json
import shutil


from detectors_evaluation.bootstrapper import DST_SHAPE


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


if __name__ == "__main__":
    target_annotations = get_target_annotations(target_object="regulatory--stop--g1")
    print(f"Total {len(target_annotations)} annotations of target class")

    dst_dir = dataset_directory / "stop_sign_images"
    copy_images_to_directory(target_annotations, dst_dir)

    dst_dir = dataset_directory / "stop_sign_images_resized"
    resize_images_and_save_to_directory(target_annotations, dst_dir)
