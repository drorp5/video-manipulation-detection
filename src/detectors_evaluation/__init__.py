from detectors_evaluation.manipulator import (
    Injector,
    FullFrameInjector,
    StripeInjector,
    SignPatchInjector,
    RectangularPatchInjector,
)
from detectors_evaluation.evaluator import (
    Evaluator,
    EvaluationResult,
    Label,
    evaluate_pair,
)
from detectors_evaluation.datasets import (
    EvaluationDataset,
    FramesDirectoryDataset,
    VideoDataset,
    VideosDirectoryDataset,
)
