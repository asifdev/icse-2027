from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


__all__ = ["create_temporal_windows"]


@dataclass(frozen=True)
class WindowBounds:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _normalize_granularity(granularity: str) -> str:
    g = str(granularity).strip().lower()

    aliases = {
        "daily": "D",
        "day": "D",
        "d": "D",
        "weekly": "W",
        "week": "W",
        "w": "W",
        "monthly": "MS",
        "month": "MS",
        "m": "MS",
        "quarterly": "QS",
        "quarter": "QS",
        "q": "QS",
    }

    return aliases.get(g, granularity)


def _period_delta(granularity: str) -> pd.DateOffset:
    g = str(granularity).strip().lower()

    if g in {"daily", "day", "d", "D"}:
        return pd.DateOffset(days=1)

    if g in {"weekly", "week", "w", "W"}:
        return pd.DateOffset(weeks=1)

    if g in {"monthly", "month", "m", "MS"}:
        return pd.DateOffset(months=1)

    if g in {"quarterly", "quarter", "q", "QS"}:
        return pd.DateOffset(months=3)

    # Fallback for pandas-style values such as "30D".
    try:
        offset = pd.tseries.frequencies.to_offset(granularity)
        return pd.DateOffset(seconds=offset.delta.total_seconds())
    except Exception:
        return pd.DateOffset(months=1)


def _compact_inputs(
    builds: pd.DataFrame,
    tests: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only columns needed for temporal window construction.

    This is critical for large datasets. The previous implementation sliced the
    full normalized_tests table per project, which caused pyarrow/pandas to
    allocate tens of GB.
    """

    build_cols = [c for c in ["project", "build_id", "build_started_at"] if c in builds.columns]
    test_cols = [c for c in ["project", "started_at", "test_count"] if c in tests.columns]

    builds = builds.loc[:, build_cols].copy()
    tests = tests.loc[:, test_cols].copy()

    if "project" not in builds.columns:
        raise ValueError("builds is missing required column: project")

    if "project" not in tests.columns:
        raise ValueError("tests is missing required column: project")

    if "build_started_at" not in builds.columns:
        raise ValueError("builds is missing required column: build_started_at")

    if "started_at" not in tests.columns:
        raise ValueError("tests is missing required column: started_at")

    builds["project"] = builds["project"].astype("string[python]")
    tests["project"] = tests["project"].astype("string[python]")

    builds["build_started_at"] = pd.to_datetime(
        builds["build_started_at"],
        errors="coerce",
        utc=True,
    )

    tests["started_at"] = pd.to_datetime(
        tests["started_at"],
        errors="coerce",
        utc=True,
    )

    builds = builds.dropna(subset=["project", "build_started_at"])
    tests = tests.dropna(subset=["project", "started_at"])

    return builds, tests


def _project_date_range(
    project_builds: pd.DataFrame,
    project_tests: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    starts = []

    if not project_builds.empty:
        starts.append(project_builds["build_started_at"].min())
        starts.append(project_builds["build_started_at"].max())

    if not project_tests.empty:
        starts.append(project_tests["started_at"].min())
        starts.append(project_tests["started_at"].max())

    starts = [x for x in starts if pd.notna(x)]

    if not starts:
        return None, None

    return min(starts), max(starts)


def _make_bounds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    granularity: str,
    strategy: str,
) -> Iterable[WindowBounds]:
    """
    Generate train/test windows.

    Supported strategies:
    - rolling
    - sliding
    - expanding
    - fixed

    For unknown strategy names, rolling behavior is used.
    """

    step = _period_delta(granularity)

    strategy_norm = str(strategy).strip().lower()

    cursor = start + step

    while cursor + step <= end + pd.Timedelta(microseconds=1):
        test_start = cursor
        test_end = cursor + step

        if strategy_norm in {"expanding", "cumulative"}:
            train_start = start
            train_end = test_start
        elif strategy_norm in {"fixed"}:
            train_start = start
            train_end = start + step
            test_start = cursor
            test_end = cursor + step
        else:
            # rolling/sliding/default: previous period trains, next period tests
            train_start = cursor - step
            train_end = cursor

        if train_start < train_end and test_start < test_end:
            yield WindowBounds(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

        cursor = cursor + step


def _count_between(
    frame: pd.DataFrame,
    time_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    if frame.empty:
        return 0

    values = frame[time_col]

    mask = (values >= start) & (values < end)

    if "test_count" in frame.columns:
        return int(pd.to_numeric(frame.loc[mask, "test_count"], errors="coerce").fillna(0).sum())

    return int(mask.sum())


def create_temporal_windows(
    builds: pd.DataFrame,
    tests: pd.DataFrame,
    granularities,
    strategies,
    min_builds_per_window: int = 1,
    min_tests_per_window: int = 1,
) -> pd.DataFrame:
    """
    Create project-level temporal train/test windows.

    Memory-safe design:
    - Drops all unnecessary columns before filtering by project.
    - Converts project columns to string[python] to avoid Arrow take().
    - Does not slice the full normalized_tests table with all columns.
    """

    builds, tests = _compact_inputs(builds, tests)

    min_builds_per_window = int(min_builds_per_window or 1)
    min_tests_per_window = int(min_tests_per_window or 1)

    if isinstance(granularities, str):
        granularities = [granularities]

    if isinstance(strategies, str):
        strategies = [strategies]

    rows = []

    projects = pd.Index(
        pd.concat(
            [
                builds["project"],
                tests["project"],
            ],
            ignore_index=True,
        )
        .dropna()
        .unique()
    ).sort_values()

    for project in projects:
        project_text = str(project)

        # These slices are now small because builds/tests contain only needed columns.
        b = builds.loc[builds["project"].eq(project), :]
        t = tests.loc[tests["project"].eq(project), :]

        project_start, project_end = _project_date_range(b, t)

        if project_start is None or project_end is None:
            continue

        for granularity in granularities:
            granularity_label = str(granularity)
            normalized_granularity = _normalize_granularity(granularity_label)

            for strategy in strategies:
                strategy_label = str(strategy)

                window_index = 0

                for bounds in _make_bounds(
                    project_start,
                    project_end,
                    normalized_granularity,
                    strategy_label,
                ):
                    num_builds_train = _count_between(
                        b,
                        "build_started_at",
                        bounds.train_start,
                        bounds.train_end,
                    )
                    num_builds_test = _count_between(
                        b,
                        "build_started_at",
                        bounds.test_start,
                        bounds.test_end,
                    )
                    num_tests_train = _count_between(
                        t,
                        "started_at",
                        bounds.train_start,
                        bounds.train_end,
                    )
                    num_tests_test = _count_between(
                        t,
                        "started_at",
                        bounds.test_start,
                        bounds.test_end,
                    )

                    if num_builds_train < min_builds_per_window:
                        continue

                    if num_builds_test < min_builds_per_window:
                        continue

                    if num_tests_train < min_tests_per_window:
                        continue

                    if num_tests_test < min_tests_per_window:
                        continue

                    window_id = (
                        f"{project_text}__{granularity_label}__"
                        f"{strategy_label}__{window_index:05d}"
                    )

                    rows.append(
                        {
                            "window_id": window_id,
                            "project": project_text,
                            "granularity": granularity_label,
                            "strategy": strategy_label,
                            "window_index": window_index,
                            "train_start": bounds.train_start,
                            "train_end": bounds.train_end,
                            "test_start": bounds.test_start,
                            "test_end": bounds.test_end,
                            "num_builds_train": num_builds_train,
                            "num_builds_test": num_builds_test,
                            "num_tests_train": num_tests_train,
                            "num_tests_test": num_tests_test,
                        }
                    )

                    window_index += 1

    columns = [
        "window_id",
        "project",
        "granularity",
        "strategy",
        "window_index",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "num_builds_train",
        "num_builds_test",
        "num_tests_train",
        "num_tests_test",
    ]

    return pd.DataFrame(rows, columns=columns)
