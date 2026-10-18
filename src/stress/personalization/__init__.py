"""User personalization module."""

from stress.personalization.centering import (
    center_features,
    compute_calibration_mean,
)
from stress.personalization.experiment import (
    SubjectConditionResult,
    aggregate_conditions,
    compute_eval_metrics,
    format_summary_table,
    prepare_centered_training,
    run_personalization_experiment,
)
from stress.personalization.splitting import (
    CalibrationSplit,
    get_calibration_windows,
    split_calibration_and_test,
)
from stress.personalization.threshold import (
    calibrated_threshold,
    predict_with_threshold,
)

__all__ = [
    "CalibrationSplit",
    "SubjectConditionResult",
    "aggregate_conditions",
    "calibrated_threshold",
    "center_features",
    "compute_calibration_mean",
    "compute_eval_metrics",
    "format_summary_table",
    "get_calibration_windows",
    "predict_with_threshold",
    "prepare_centered_training",
    "run_personalization_experiment",
    "split_calibration_and_test",
]
