from __future__ import annotations

import numpy as np
import pandas as pd

from ci_driftlab.preprocessing.normalize import normalize_outcome


__all__ = ["extract_features"]

SECONDS_PER_DAY = 86400.0


def _as_timestamp(value):
    return pd.to_datetime(value, errors="coerce", utc=True)


def _force_python_backing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive conversion away from Arrow-backed columns.

    The LRTS dataset is large enough that boolean slicing on Arrow-backed
    columns can request tens of GB during DataFrame.take().
    """
    df = df.copy(deep=False)

    for col in df.columns:
        dtype_text = str(df[col].dtype).lower()
        if "pyarrow" in dtype_text or dtype_text.endswith("[pyarrow]"):
            df[col] = df[col].astype("object")

    return df


def _prepare_windows(windows: pd.DataFrame) -> pd.DataFrame:
    windows = _force_python_backing(windows)

    required = [
        "window_id",
        "project",
        "granularity",
        "strategy",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ]

    missing = [c for c in required if c not in windows.columns]
    if missing:
        raise ValueError(f"project_windows.csv missing required columns: {missing}")

    windows = windows.copy()
    windows["project"] = windows["project"].astype("string[python]")

    for col in ["train_start", "train_end", "test_start", "test_end"]:
        windows[col] = pd.to_datetime(windows[col], errors="coerce", utc=True)

    windows = windows.dropna(
        subset=[
            "project",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        ]
    )

    return windows.reset_index(drop=True)


def _prepare_tests(tests: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns needed for feature extraction.

    This is the main memory fix. The old implementation sliced the full
    normalized_tests table for every window.
    """

    candidate_cols = [
        "project",
        "build_id",
        "test_id",
        "test_name",
        "test_class",
        "started_at",
        "duration",
        "duration_sec",
        "is_failed",
        "test_status",
        "status",
        "outcome",
    ]

    cols = [c for c in candidate_cols if c in tests.columns]
    tests = tests.loc[:, cols].copy()
    tests = _force_python_backing(tests)

    required = ["project", "started_at"]
    missing = [c for c in required if c not in tests.columns]
    if missing:
        raise ValueError(f"normalized_tests.csv missing required columns: {missing}")

    if "test_id" not in tests.columns and "test_name" in tests.columns:
        tests["test_id"] = tests["test_name"]

    if "test_name" not in tests.columns and "test_id" in tests.columns:
        tests["test_name"] = tests["test_id"]

    if "test_id" not in tests.columns and "test_class" in tests.columns:
        tests["test_id"] = tests["test_class"]

    if "test_name" not in tests.columns and "test_class" in tests.columns:
        tests["test_name"] = tests["test_class"]

    if "test_id" in tests.columns and "test_name" in tests.columns:
        missing = tests["test_id"].isna()
        if bool(missing.any()):
            tests.loc[missing, "test_id"] = tests.loc[missing, "test_name"]
        elif len(tests) <= 5_000_000:
            text = tests["test_id"].astype("object")
            missing = text.map(lambda value: str(value).strip() in {"", "<NA>", "nan", "None"})
            if bool(missing.any()):
                tests.loc[missing, "test_id"] = tests.loc[missing, "test_name"]

    if "test_id" not in tests.columns:
        tests["test_id"] = "unknown_test"

    if "test_name" not in tests.columns:
        tests["test_name"] = tests["test_id"]

    tests["project"] = tests["project"].astype("string[python]")
    tests["test_id"] = tests["test_id"].astype("string[python]")
    tests["test_name"] = tests["test_name"].astype("string[python]")

    tests["started_at"] = pd.to_datetime(
        tests["started_at"],
        errors="coerce",
        utc=True,
    )

    if "duration" not in tests.columns and "duration_sec" in tests.columns:
        tests["duration"] = tests["duration_sec"]

    if "duration" in tests.columns:
        tests["duration"] = pd.to_numeric(tests["duration"], errors="coerce")
    else:
        tests["duration"] = np.nan

    failure_sources = []

    if "is_failed" in tests.columns:
        failure_sources.append(tests["is_failed"].map(normalize_outcome).eq("fail"))

    for col in ["test_status", "status", "outcome"]:
        if col in tests.columns:
            failure_sources.append(tests[col].map(normalize_outcome).eq("fail"))

    if failure_sources:
        is_failed = failure_sources[0].copy()
        for source in failure_sources[1:]:
            is_failed = is_failed | source
        tests["is_failed"] = is_failed.fillna(False).astype(bool)
    else:
        raise ValueError(
            "normalized_tests.csv must contain one of: is_failed, test_status, status, outcome. "
            "Refusing to assume all tests passed."
        )

    tests = tests.dropna(subset=["project", "started_at"])
    return tests.reset_index(drop=True)


def _prepare_builds(builds: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        "project",
        "build_id",
        "build_started_at",
        "status",
        "branch",
        "commit_sha",
    ]

    cols = [c for c in candidate_cols if c in builds.columns]
    builds = builds.loc[:, cols].copy()
    builds = _force_python_backing(builds)

    if "project" not in builds.columns:
        raise ValueError("normalized_builds.csv missing required column: project")

    builds["project"] = builds["project"].astype("string[python]")

    if "build_started_at" in builds.columns:
        builds["build_started_at"] = pd.to_datetime(
            builds["build_started_at"],
            errors="coerce",
            utc=True,
        )
    else:
        raise ValueError("normalized_builds.csv missing required column: build_started_at")

    builds = builds.dropna(subset=["project", "build_started_at"])
    return builds.reset_index(drop=True)


def _safe_mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()
    return float(value) if pd.notna(value) else float("nan")


def _safe_median(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").median()
    return float(value) if pd.notna(value) else float("nan")


def _safe_sum_bool(series: pd.Series) -> int:
    if series.empty:
        return 0
    return int(series.fillna(False).astype(bool).sum())


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return float("nan")
    return float(numerator) / denominator


def _window_days(start, end) -> float:
    start = _as_timestamp(start)
    end = _as_timestamp(end)

    if pd.isna(start) or pd.isna(end) or end <= start:
        return float("nan")

    return float((end - start).total_seconds() / SECONDS_PER_DAY)


def _cadence(count: int, days: float) -> float:
    if pd.isna(days) or days <= 0:
        return float("nan")
    return float(count) / float(days)


def _slice_project_time(
    project_frame: pd.DataFrame,
    time_col: str,
    start,
    end,
) -> pd.DataFrame:
    """
    Slice only a single project's already-reduced frame.

    This avoids slicing the full LRTS table per window.
    """
    if project_frame.empty:
        return project_frame

    start = _as_timestamp(start)
    end = _as_timestamp(end)

    if pd.isna(start) or pd.isna(end):
        return project_frame.iloc[0:0]

    mask = project_frame[time_col].ge(start) & project_frame[time_col].lt(end)
    return project_frame.loc[mask]


def _project_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = {}

    if df.empty or "project" not in df.columns:
        return groups

    for project, group in df.groupby("project", sort=False, observed=True):
        groups[str(project)] = group.reset_index(drop=True)

    return groups


def _test_id_set(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "test_id" not in frame.columns:
        return set()
    return set(frame["test_id"].dropna().astype(str))


def _window_feature_row(
    dataset: str,
    w,
    train_tests: pd.DataFrame,
    test_tests: pd.DataFrame,
    train_builds: pd.DataFrame,
    test_builds: pd.DataFrame,
) -> dict:
    train_count = int(len(train_tests))
    test_count = int(len(test_tests))
    train_build_count = int(len(train_builds))
    test_build_count = int(len(test_builds))

    train_days = _window_days(w.train_start, w.train_end)
    test_days = _window_days(w.test_start, w.test_end)

    train_build_cadence = _cadence(train_build_count, train_days)
    test_build_cadence = _cadence(test_build_count, test_days)
    build_cadence = test_build_cadence

    train_failed = (
        _safe_sum_bool(train_tests["is_failed"])
        if "is_failed" in train_tests.columns
        else 0
    )
    test_failed = (
        _safe_sum_bool(test_tests["is_failed"])
        if "is_failed" in test_tests.columns
        else 0
    )

    unique_train_tests = (
        int(train_tests["test_id"].nunique())
        if "test_id" in train_tests.columns and not train_tests.empty
        else 0
    )

    unique_test_tests = (
        int(test_tests["test_id"].nunique())
        if "test_id" in test_tests.columns and not test_tests.empty
        else 0
    )

    train_test_ids = _test_id_set(train_tests)
    test_test_ids = _test_id_set(test_tests)

    test_union = train_test_ids | test_test_ids
    test_intersection = train_test_ids & test_test_ids

    test_churn = (
        1.0 - (len(test_intersection) / len(test_union))
        if test_union
        else float("nan")
    )

    new_test_ratio = (
        len(test_test_ids - train_test_ids) / len(test_test_ids)
        if test_test_ids
        else float("nan")
    )

    retired_test_ratio = (
        len(train_test_ids - test_test_ids) / len(train_test_ids)
        if train_test_ids
        else float("nan")
    )

    train_failure_rate = _safe_rate(train_failed, train_count)
    test_failure_rate = _safe_rate(test_failed, test_count)

    mean_train_duration = (
        _safe_mean(train_tests["duration"])
        if "duration" in train_tests.columns and not train_tests.empty
        else float("nan")
    )
    median_train_duration = (
        _safe_median(train_tests["duration"])
        if "duration" in train_tests.columns and not train_tests.empty
        else float("nan")
    )
    mean_test_duration = (
        _safe_mean(test_tests["duration"])
        if "duration" in test_tests.columns and not test_tests.empty
        else float("nan")
    )
    median_test_duration = (
        _safe_median(test_tests["duration"])
        if "duration" in test_tests.columns and not test_tests.empty
        else float("nan")
    )

    # Detector-compatible window features for software-aware Phase 2 detectors.
    failure_rate = test_failure_rate
    duration_mean = mean_test_duration
    duration_median = median_test_duration
    suite_size = unique_test_tests

    return {
        "dataset": dataset,
        "window_id": w.window_id,
        "project": w.project,
        "granularity": w.granularity,
        "strategy": w.strategy,
        "train_start": w.train_start,
        "train_end": w.train_end,
        "test_start": w.test_start,
        "test_end": w.test_end,
        "train_days": train_days,
        "test_days": test_days,
        "num_train_tests": train_count,
        "num_test_tests": test_count,
        "num_train_builds": train_build_count,
        "num_test_builds": test_build_count,
        "train_build_cadence": train_build_cadence,
        "test_build_cadence": test_build_cadence,
        "build_cadence": build_cadence,
        "unique_train_tests": unique_train_tests,
        "unique_test_tests": unique_test_tests,
        "suite_size": suite_size,
        "test_churn": test_churn,
        "new_test_ratio": new_test_ratio,
        "retired_test_ratio": retired_test_ratio,
        "train_failures": train_failed,
        "test_failures": test_failed,
        "train_failure_rate": train_failure_rate,
        "test_failure_rate": test_failure_rate,
        "failure_rate": failure_rate,
        "mean_train_duration": mean_train_duration,
        "median_train_duration": median_train_duration,
        "mean_test_duration": mean_test_duration,
        "median_test_duration": median_test_duration,
        "duration_mean": duration_mean,
        "duration_median": duration_median,
    }


def _test_feature_rows(
    dataset: str,
    w,
    train_tests: pd.DataFrame,
    test_tests: pd.DataFrame,
) -> list[dict]:
    """
    Per-test feature rows for each window.

    This intentionally summarizes by test_id instead of carrying full raw rows.
    """

    if train_tests.empty and test_tests.empty:
        return []

    rows = []
    train_grouped = {}
    test_grouped = {}

    if not train_tests.empty:
        for test_id, group in train_tests.groupby("test_id", sort=False, observed=True):
            train_grouped[str(test_id)] = group

    if not test_tests.empty:
        for test_id, group in test_tests.groupby("test_id", sort=False, observed=True):
            test_grouped[str(test_id)] = group

    all_test_ids = sorted(set(train_grouped) | set(test_grouped))

    for test_id in all_test_ids:
        tr = train_grouped.get(test_id)
        te = test_grouped.get(test_id)

        if tr is None:
            tr = train_tests.iloc[0:0]

        if te is None:
            te = test_tests.iloc[0:0]

        tr_count = int(len(tr))
        te_count = int(len(te))

        tr_failures = (
            _safe_sum_bool(tr["is_failed"])
            if "is_failed" in tr.columns
            else 0
        )
        te_failures = (
            _safe_sum_bool(te["is_failed"])
            if "is_failed" in te.columns
            else 0
        )

        test_name = test_id
        if "test_name" in tr.columns and not tr.empty:
            test_name = str(tr["test_name"].iloc[0])
        elif "test_name" in te.columns and not te.empty:
            test_name = str(te["test_name"].iloc[0])

        rows.append(
            {
                "dataset": dataset,
                "window_id": w.window_id,
                "project": w.project,
                "granularity": w.granularity,
                "strategy": w.strategy,
                "test_id": test_id,
                "test_name": test_name,
                "train_executions": tr_count,
                "test_executions": te_count,
                "train_failures": tr_failures,
                "test_failures": te_failures,
                "train_failure_rate": _safe_rate(tr_failures, tr_count),
                "test_failure_rate": _safe_rate(te_failures, te_count),
                "mean_duration": (
                    _safe_mean(tr["duration"])
                    if "duration" in tr.columns and not tr.empty
                    else float("nan")
                ),
                "median_duration": (
                    _safe_median(tr["duration"])
                    if "duration" in tr.columns and not tr.empty
                    else float("nan")
                ),
                "is_failed": bool(te_failures > 0),
            }
        )

    return rows


def extract_features(
    windows: pd.DataFrame,
    builds: pd.DataFrame,
    tests: pd.DataFrame,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract window-level and test-level features.

    Memory-safe design:
    - Reduces tests/builds to only needed columns.
    - Splits data by project once.
    - Slices only per-project frames per window.
    - Uses half-open time windows: [start, end).

    Detector-compatible output columns include:
    - failure_rate
    - duration_mean
    - duration_median
    - suite_size
    - test_churn
    - build_cadence
    """

    windows = _prepare_windows(windows)
    builds = _prepare_builds(builds)
    tests = _prepare_tests(tests)

    tests_by_project = _project_groups(tests)
    builds_by_project = _project_groups(builds)

    window_rows = []
    test_feature_rows = []

    for w in windows.itertuples(index=False):
        project = str(w.project)

        project_tests = tests_by_project.get(project)
        if project_tests is None:
            project_tests = tests.iloc[0:0]

        project_builds = builds_by_project.get(project)
        if project_builds is None:
            project_builds = builds.iloc[0:0]

        train_tests = _slice_project_time(
            project_tests,
            "started_at",
            w.train_start,
            w.train_end,
        )

        test_tests = _slice_project_time(
            project_tests,
            "started_at",
            w.test_start,
            w.test_end,
        )

        train_builds = _slice_project_time(
            project_builds,
            "build_started_at",
            w.train_start,
            w.train_end,
        )

        test_builds = _slice_project_time(
            project_builds,
            "build_started_at",
            w.test_start,
            w.test_end,
        )

        window_rows.append(
            _window_feature_row(
                dataset,
                w,
                train_tests,
                test_tests,
                train_builds,
                test_builds,
            )
        )

        test_feature_rows.extend(
            _test_feature_rows(
                dataset,
                w,
                train_tests,
                test_tests,
            )
        )

    window_columns = [
        "dataset",
        "window_id",
        "project",
        "granularity",
        "strategy",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "train_days",
        "test_days",
        "num_train_tests",
        "num_test_tests",
        "num_train_builds",
        "num_test_builds",
        "train_build_cadence",
        "test_build_cadence",
        "build_cadence",
        "unique_train_tests",
        "unique_test_tests",
        "suite_size",
        "test_churn",
        "new_test_ratio",
        "retired_test_ratio",
        "train_failures",
        "test_failures",
        "train_failure_rate",
        "test_failure_rate",
        "failure_rate",
        "mean_train_duration",
        "median_train_duration",
        "mean_test_duration",
        "median_test_duration",
        "duration_mean",
        "duration_median",
    ]

    test_columns = [
        "dataset",
        "window_id",
        "project",
        "granularity",
        "strategy",
        "test_id",
        "test_name",
        "train_executions",
        "test_executions",
        "train_failures",
        "test_failures",
        "train_failure_rate",
        "test_failure_rate",
        "mean_duration",
        "median_duration",
        "is_failed",
    ]

    return (
        pd.DataFrame(window_rows, columns=window_columns),
        pd.DataFrame(test_feature_rows, columns=test_columns),
    )
