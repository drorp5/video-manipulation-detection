import sys

from scipy.fft import dst

sys.path.append("./src")
from pathlib import Path
import cv2
from tqdm import tqdm
import multiprocessing
from functools import partial
import pandas as pd
from manipulation_detectors.image_processing import (
    OpticalFlowDetector,
    MSEImageDetector,
    HueSaturationHistogramDetector,
)
from detectors_evaluation import (
    FullFrameInjector,
    StripeInjector,
    SignPatchInjector,
    Evaluator,
    EvaluationResult,
    EvaluationDataset,
    FramesDirectoryDataset,
    Label,
    evaluate_pair,
    VideoDataset,
    VideosDirectoryDataset,
)
import detectors_evaluation.bootstrapper as bootstrapper


def run(evaluator: Evaluator, dataset: EvaluationDataset, dst_dir_path: Path):
    res_by_detector = {
        detector.name: {"score": [], "label": []} for detector in evaluator.detectors
    }

    helper = partial(evaluate_pair, evaluator)
    with multiprocessing.Pool(6) as pool:
        for res in tqdm(pool.imap(helper, dataset), total=len(dataset)):
            for detector_res in res:
                res_by_detector[detector_res.detector]["label"].append(Label.REAL.value)
                res_by_detector[detector_res.detector]["score"].append(
                    detector_res.real
                )
                res_by_detector[detector_res.detector]["label"].append(Label.FAKE.value)
                res_by_detector[detector_res.detector]["score"].append(
                    detector_res.fake
                )

        # all_res = []
        # for (frame_1, frame_2) in tqdm(dataset):
        #     res = evaluator.evaluate(frame_1, frame_2)
        #     all_res.append(res)

        # save results
        for detector in evaluator.detectors:
            df = pd.DataFrame(res_by_detector[detector.name])
            dst_path = (
                dst_dir_path
                / f"{dataset.name}_{evaluator.injector.name}_{detector.name}.csv"
            )
            df.to_csv(dst_path)


if __name__ == "__main__":
    base_dir = Path("OUTPUT")
    only_save_example = False
    name = "fake_red_light"

    dst_dir = base_dir / name
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)

    # set detectors
    detectors = bootstrapper.get_image_processing_detectors()

    for data_source in ["BDD"]:  # "experiment" ,"BDD"]:
        for injector_type in ["full_frame", "stripe", "patch"]:
            print(f"running {injector_type} on {data_source}")
            # set dataset
            dataset = bootstrapper.get_dataset(data_source)
            # injector = bootstrapper.get_stop_sign_injector(injector_type)
            injector = bootstrapper.get_red_light_injector(injector_type)
            # run
            if only_save_example:
                # save example
                dst_path = dst_dir / f"fake_{data_source}_{injector.name}.jpg"
                fake_frame = injector.inject(
                    frame_1=dataset[0][0], frame_2=dataset[0][1]
                )
                fake_frame_bgr = cv2.cvtColor(fake_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_path.as_posix(), fake_frame_bgr)
                print(f"saved in {dst_path}")
            else:
                evaluator = Evaluator(detectors=detectors, injector=injector)
                run(evaluator=evaluator, dataset=dataset, dst_dir_path=dst_dir)
