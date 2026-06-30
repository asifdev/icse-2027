from __future__ import annotations

from pathlib import Path

import pandas as pd

from ci_driftlab.io_utils import write_csv
from ci_driftlab.drift_detectors.generic import run_generic_detectors
from ci_driftlab.drift_detectors.software_aware import run_software_aware_detectors
from ci_driftlab.drift_detectors.hybrid import combine_detectors
from ci_driftlab.taxonomy.drift_taxonomy import assign_taxonomy
from ci_driftlab.metrics.drift_metrics import evaluate_detector_frame


def _force_python_backing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Avoid Arrow-backed pandas columns in Phase 2.

    Phase 2 usually works on smaller aggregate files, but this keeps behavior
    consistent with the Phase 1 memory-safe patches.
    """

    df = df.copy(deep=False)

    for col in df.columns:
        dtype_text = str(df[col].dtype).lower()
        if "pyarrow" in dtype_text or dtype_text.endswith("[pyarrow]"):
            df[col] = df[col].astype("object")

    return df


def safe_read_csv(path: Path, usecols=None) -> pd.DataFrame:
    path = Path(path)

    if usecols:
        wanted = set(usecols)
        df = pd.read_csv(
            path,
            usecols=lambda c: c in wanted,
            low_memory=False,
        )
    else:
        df = pd.read_csv(path, low_memory=False)

    df = _force_python_backing(df)

    for col in [
        "dataset",
        "project",
        "window_id",
        "granularity",
        "strategy",
        "detector",
        "detector_family",
        "model",
        "policy",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string[python]")

    return df


def _copy_alias(df: pd.DataFrame, target: str, sources: list[str]) -> pd.DataFrame:
    """
    Add target column from the first available source column.

    This keeps older detector modules working even when Phase 1 uses clearer
    train/test-prefixed feature names.
    """

    if target in df.columns:
        return df

    for source in sources:
        if source in df.columns:
            df[target] = df[source]
            return df

    return df


def add_feature_aliases(features: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible aliases for software-aware detectors.

    Older detector code expects names such as:
    - failure_rate
    - num_tests
    - num_builds
    - failures
    - mean_duration

    The patched Phase 1 produces clearer names such as:
    - test_failure_rate
    - num_test_tests
    - num_test_builds
    - test_failures
    - mean_test_duration
    """

    features = features.copy()

    alias_map = {
        "failure_rate": [
            "test_failure_rate",
            "train_failure_rate",
        ],
        "train_rate": [
            "train_failure_rate",
        ],
        "test_rate": [
            "test_failure_rate",
        ],
        "num_tests": [
            "num_test_tests",
            "unique_test_tests",
            "num_train_tests",
        ],
        "num_builds": [
            "num_test_builds",
            "num_train_builds",
        ],
        "failures": [
            "test_failures",
            "train_failures",
        ],
        "failure_count": [
            "test_failures",
            "train_failures",
        ],
        "mean_duration": [
            "mean_test_duration",
            "mean_train_duration",
        ],
        "median_duration": [
            "median_test_duration",
            "median_train_duration",
        ],
        "unique_tests": [
            "unique_test_tests",
            "unique_train_tests",
        ],
        "duration_mean": [
            "mean_test_duration",
            "mean_train_duration",
            "mean_duration",
        ],

        "duration_median": [
            "median_test_duration",
            "median_train_duration",
            "median_duration",
        ],

        "duration": [
            "mean_test_duration",
            "mean_train_duration",
            "mean_duration",
        ],
        "suite_size": [
            "num_test_tests",
            "unique_test_tests",
            "num_train_tests",
            "unique_train_tests",
            "num_tests",
        ]
    }

    for target, sources in alias_map.items():
        features = _copy_alias(features, target, sources)

    return features


def add_test_feature_aliases(test_features: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible aliases for per-test generic drift detectors.

    Phase 1's current per-test export uses explicit train/test names such as
    test_failure_rate and mean_duration. Generic distribution detectors expect
    the older logical feature names.
    """

    test_features = test_features.copy()

    alias_map = {
        "failure_rate": [
            "test_failure_rate",
            "train_failure_rate",
        ],
        "recent_failure_rate": [
            "test_failure_rate",
            "failure_rate",
            "train_failure_rate",
        ],
        "failure_count": [
            "test_failures",
            "train_failures",
        ],
        "duration_mean": [
            "mean_duration",
            "duration_mean",
            "median_duration",
        ],
        "duration_std": [
            "std_duration",
            "duration_std",
        ],
        "test_age": [
            "test_age",
            "train_executions",
            "test_executions",
        ],
        "last_failure_age": [
            "last_failure_age",
        ],
    }

    for target, sources in alias_map.items():
        test_features = _copy_alias(test_features, target, sources)

    return test_features


def add_baseline_aliases(baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible aliases for baseline/TCP performance columns.
    """

    baseline = baseline.copy()

    alias_map = {
        "apfd": ["APFD"],
        "apfdc": ["APFDc"],
        "napfd": ["NAPFD"],
        "num_failures": ["failures", "test_failures"],
    }

    for target, sources in alias_map.items():
        baseline = _copy_alias(baseline, target, sources)

    return baseline


def _require_columns(df: pd.DataFrame, columns: list[str], name: str):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def run_phase2(config, dataset, out, logger, checkpoint=None):
    root = Path(out)
    p1 = root / "phase1_foundation"
    target = root / "phase2_drift_detection"
    target.mkdir(parents=True, exist_ok=True)

    dc = config["drift_detectors"]
    thresholds = dc["thresholds"]

    def execute(name, outputs, fn, inputs=(), parameters=None):
        if checkpoint:
            return checkpoint.run_step(
                name,
                list(outputs),
                fn,
                list(inputs),
                parameters or {},
            )
        return fn()

    generic_path = target / "generic_detector_results.csv"
    software_path = target / "software_aware_detector_results.csv"
    hybrid_path = target / "hybrid_detector_results.csv"

    def generic_step():
        test_features = safe_read_csv(p1 / "test_features_by_window.csv")
        features = safe_read_csv(p1 / "features_by_window.csv")
        test_features = add_test_feature_aliases(test_features)
        features = add_feature_aliases(features)

        result = run_generic_detectors(
            test_features,
            features,
            thresholds,
            dc["generic"],
        )

        write_csv(result, generic_path)

    execute(
        "phase2.generic_detectors",
        [generic_path],
        generic_step,
        [
            p1 / "test_features_by_window.csv",
            p1 / "features_by_window.csv",
        ],
        {
            "detectors": dc["generic"],
            "thresholds": thresholds,
        },
    )

    def software_step():
        features = safe_read_csv(p1 / "features_by_window.csv")
        baseline = safe_read_csv(p1 / "baseline_tcp_results.csv")

        features = add_feature_aliases(features)
        baseline = add_baseline_aliases(baseline)

        _require_columns(
            features,
            [
                "window_id",
                "project",
            ],
            "features_by_window.csv",
        )

        _require_columns(
            baseline,
            [
                "window_id",
                "project",
            ],
            "baseline_tcp_results.csv",
        )

        result = run_software_aware_detectors(
            features,
            baseline,
            thresholds,
            dc["software_aware"],
        )

        write_csv(result, software_path)

    execute(
        "phase2.software_detectors",
        [software_path],
        software_step,
        [
            p1 / "features_by_window.csv",
            p1 / "baseline_tcp_results.csv",
        ],
        {
            "detectors": dc["software_aware"],
            "thresholds": thresholds,
        },
    )

    def hybrid_step():
        generic = safe_read_csv(generic_path)
        software = safe_read_csv(software_path)

        result = combine_detectors(
            generic,
            software,
        )

        write_csv(result, hybrid_path)

    execute(
        "phase2.hybrid_detectors",
        [hybrid_path],
        hybrid_step,
        [
            generic_path,
            software_path,
        ],
    )

    eval_path = target / "detector_precision_recall_delay.csv"

    def evaluation_step():
        generic = safe_read_csv(generic_path)
        software = safe_read_csv(software_path)

        _require_columns(
            software,
            ["detector"],
            "software_aware_detector_results.csv",
        )

        truth = software[software["detector"].astype(str).eq("performance_degradation")]

        if truth.empty:
            logger.warning(
                "No detector == 'performance_degradation' rows found in "
                "software_aware_detector_results.csv. Evaluation may be empty."
            )

        software_without_truth = software[
            ~software["detector"].astype(str).eq("performance_degradation")
        ]

        evaluation = pd.concat(
            [
                evaluate_detector_frame(
                    generic,
                    truth,
                    "generic",
                ),
                evaluate_detector_frame(
                    software_without_truth,
                    truth,
                    "software_aware",
                ),
            ],
            ignore_index=True,
        )

        write_csv(evaluation, eval_path)

    execute(
        "phase2.evaluation",
        [eval_path],
        evaluation_step,
        [
            generic_path,
            software_path,
        ],
    )

    sensitivity_path = target / "detector_threshold_sensitivity.csv"

    def sensitivity_step():
        sensitivity = []
        evaluation = safe_read_csv(eval_path)

        if evaluation.empty:
            result = pd.DataFrame(
                columns=[
                    "detector_family",
                    "detector",
                    "threshold_multiplier",
                    "baseline_precision",
                    "baseline_recall",
                    "notes",
                ]
            )
            write_csv(result, sensitivity_path)
            return

        required = [
            "detector_family",
            "detector",
            "precision",
            "recall",
        ]

        missing = [c for c in required if c not in evaluation.columns]
        if missing:
            raise ValueError(
                f"detector_precision_recall_delay.csv missing required columns: {missing}"
            )

        for multiplier in config["cost_model"]["sensitivity_multipliers"]:
            for row in evaluation.itertuples(index=False):
                sensitivity.append(
                    {
                        "detector_family": row.detector_family,
                        "detector": row.detector,
                        "threshold_multiplier": multiplier,
                        "baseline_precision": row.precision,
                        "baseline_recall": row.recall,
                        "notes": (
                            "reported baseline metrics; rerun detector threshold "
                            "sweep in full analysis"
                        ),
                    }
                )

        write_csv(
            pd.DataFrame(
                sensitivity,
                columns=[
                    "detector_family",
                    "detector",
                    "threshold_multiplier",
                    "baseline_precision",
                    "baseline_recall",
                    "notes",
                ],
            ),
            sensitivity_path,
        )

    execute(
        "phase2.sensitivity",
        [sensitivity_path],
        sensitivity_step,
        [eval_path],
        {
            "multipliers": config["cost_model"]["sensitivity_multipliers"],
        },
    )

    taxonomy_outputs = [
        target / "taxonomy_labels_by_window.csv",
        target / "taxonomy_prevalence.csv",
        target / "drift_to_degradation.csv",
    ]

    def taxonomy_step():
        generic = safe_read_csv(generic_path)
        software = safe_read_csv(software_path)

        labels, prevalence, relation = assign_taxonomy(
            generic,
            software,
        )

        write_csv(labels, taxonomy_outputs[0])
        write_csv(prevalence, taxonomy_outputs[1])
        write_csv(relation, taxonomy_outputs[2])

    execute(
        "phase2.taxonomy",
        taxonomy_outputs,
        taxonomy_step,
        [
            generic_path,
            software_path,
        ],
    )

    logger.info("Phase 2 complete (checkpoint-aware)")
