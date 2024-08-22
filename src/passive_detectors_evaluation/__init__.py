from injectors.full_frame_injector import FullFrameInjector
from injectors.injector import Injector
from injectors.stop_sign_injector import SignPatchInjector
from injectors.stripe_injector import StripeInjector
from injectors.rectangular_patch_injector import (
    RectangularPatchInjector,
)
from passive_detectors_evaluation.evaluator import (
    Evaluator,
    EvaluationResult,
    Label,
    evaluate_pair,
)
from passive_detectors_evaluation.datasets import (
    EvaluationDataset,
    FramesDirectoryDataset,
    VideoDataset,
    VideosDirectoryDataset,
)
