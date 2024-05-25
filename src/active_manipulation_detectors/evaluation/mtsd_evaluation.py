from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt


from detectors_evaluation.bootstrapper import DST_SHAPE
from sign_detectors import StopSignDetector, get_detector, draw_bounding_boxes
from gige.utils import bgr_img_to_packets_payload, payload_gvsp_bytes_to_raw_image
from utils.image_processing import bggr_to_rggb

dataset_directory = Path("../datasets/mtsd_v2_fully_annotated")
original_images_directory = dataset_directory / "images"
stop_sign_images_directory = dataset_directory / "MobileNet_detections"
annotations_directory = dataset_directory / "annotations"


detector = get_detector("MobileNet")

injections_images_directory = dataset_directory / "MobileNet_detections_injections"
if not injections_images_directory.exists():
    injections_images_directory.mkdir(parents=True)


if __name__ == "__main__":
    num_symbols = 8
    max_payload_bytes = 8963

    max_width = 1936
    widths = [max_width - 2 * i for i in range(1, num_symbols)]
    for img_path in tqdm(list(stop_sign_images_directory.glob("*jpg"))):
        img_key = img_path.stem
        # read image
        img_bgr = cv2.imread(img_path.as_posix())

        # split to gvsp payloab bytes
        gvsp_payload = bgr_img_to_packets_payload(img_bgr, max_payload_bytes)

        # select stripe starting from the top
        gvsp_payload_stripe = gvsp_payload[:-5]

        # combine to image in desired width
        for width in widths:
            raw_image, assigned_pixels = payload_gvsp_bytes_to_raw_image(
                payload_packets=gvsp_payload_stripe, shape=(1216, width)
            )
            raw_image = bggr_to_rggb(raw_image)
            assigned_pixels = bggr_to_rggb(assigned_pixels)
            rgb_img = cv2.cvtColor(raw_image, cv2.COLOR_BayerRG2RGB)

            # save image
            dst_path = injections_images_directory / f"{img_key}_width_{int(width)}.jpg"
            cv2.imwrite(dst_path.as_posix(), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

            # run detection
            detections = detector.detect(rgb_img)
            if len(detections) > 0:
                print(dst_path)
                img_with_detections = draw_bounding_boxes(rgb_img, detections)
                dst_path = (
                    injections_images_directory
                    / f"{img_key}_width_{int(width)}_detected.jpg"
                )
                cv2.imwrite(
                    dst_path.as_posix(),
                    cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR),
                )
