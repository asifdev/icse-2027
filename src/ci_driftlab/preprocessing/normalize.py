from __future__ import annotations

import gc
import hashlib
from typing import Iterable

import pandas as pd
from pandas.api.types import is_bool_dtype


__all__ = [
    "FAILURE_VALUES",
    "normalize_outcome",
    "normalize_duration",
    "finalize_tables",
]

FAILURE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "fail",
    "failed",
    "failure",
    "error",
    "errored",
    "broken",
    "red",
    "regression",
    "timeout",
    "timedout",
    "timed_out",
    "timed out",
}


def normalize_outcome(value):
    """
    Backward-compatible outcome normalizer.

    Returns:
    - "fail"
    - "pass"
    - "skip"
    - "unknown"

    This keeps compatibility with the earlier CI-DriftLab code where downstream
    modules may expect compact labels instead of "failed"/"passed"/"skipped".
    """

    if pd.isna(value):
        return "unknown"

    if isinstance(value, bool):
        return "fail" if value else "pass"

    s = str(value).strip().lower()

    if s in FAILURE_VALUES:
        return "fail"

    if s in {
        "pass",
        "passed",
        "success",
        "successful",
        "ok",
        "0",
        "false",
        "green",
    }:
        return "pass"

    if s in {
        "skip",
        "skipped",
        "ignored",
        "pending",
        "disabled",
    }:
        return "skip"

    return "unknown"


def normalize_duration(value):
    """
    Normalize duration values to seconds.

    Accepts examples such as:
    - 1.5
    - "1.5"
    - "250ms"
    - "2s"
    - "3m"

    Invalid or negative durations become NaN.
    """

    if pd.isna(value):
        return float("nan")

    if isinstance(value, (int, float)):
        value = float(value)
        return value if value >= 0 else float("nan")

    s = str(value).strip().lower()

    try:
        if s.endswith("ms"):
            value = float(s[:-2]) / 1000.0
        elif s.endswith("s"):
            value = float(s[:-1])
        elif s.endswith("m"):
            value = float(s[:-1]) * 60.0
        else:
            value = float(s)

        return value if value >= 0 else float("nan")

    except (ValueError, TypeError):
        return float("nan")


def _stable(prefix: str, values: Iterable[Iterable[object]]) -> list[str]:
    """
    Create stable IDs from row-like values.

    Kept for compatibility with older code that may import/use this helper.
    """

    return [
        prefix + hashlib.sha1("|".join(map(str, row)).encode("utf-8")).hexdigest()[:16]
        for row in values
    ]


def _materialize_non_arrow(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Convert Arrow-backed columns to pandas/Python-backed columns.

    This avoids pyarrow ChunkedArray.take() during pandas operations such as
    sort_values(), drop_duplicates(), and reindexing on large CI tables.
    """

    if df is None:
        return pd.DataFrame()

    out = df.copy(deep=False)

    for col in out.columns:
        dtype_text = str(out[col].dtype).lower()

        if "pyarrow" in dtype_text or dtype_text.endswith("[pyarrow]"):
            out[col] = out[col].astype("object")

    return out


def _stringify_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert selected existing columns to Python string dtype.

    Python string dtype is used instead of Arrow string dtype to avoid Arrow
    memory spikes during downstream pandas operations.
    """

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string[python]")

    return df


def _to_bool(series: pd.Series) -> pd.Series:
    """
    Convert outcome-like values to boolean failure labels.

    True means failed.
    False means not failed.
    """

    if is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return series.map(normalize_outcome).eq("fail")


def _dropna_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]

    if not existing:
        return df

    return df.dropna(subset=existing)


def _deduplicate_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Deduplicate using only stable identity columns when available.

    Avoid full-row drop_duplicates() on large CI tables because it is expensive
    and can trigger large memory allocations.
    """

    existing = [col for col in columns if col in df.columns]

    if not existing:
        return df.drop_duplicates(keep="first")

    return df.drop_duplicates(subset=existing, keep="first")


def _ensure_test_identity_columns(tests: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure both test_id and test_name exist where possible.

    Some loaders provide test_id, some provide test_name. Downstream code in
    this project expects both in different places.
    """

    if "test_id" not in tests.columns and "test_name" in tests.columns:
        tests["test_id"] = tests["test_name"]

    if "test_name" not in tests.columns and "test_id" in tests.columns:
        tests["test_name"] = tests["test_id"]

    if "test_id" not in tests.columns and "test_class" in tests.columns:
        tests["test_id"] = tests["test_class"]

    if "test_name" not in tests.columns and "test_class" in tests.columns:
        tests["test_name"] = tests["test_class"]

    def _fill_missing_identity(target: str, source: str) -> None:
        missing = tests[target].isna()
        if bool(missing.any()):
            tests.loc[missing, target] = tests.loc[missing, source]
            return

        # Blank-string repair is useful for small/medium adapters, but a full
        # LRTS table can be >100M rows. Avoid a second full-column Python scan
        # when null detection already proved the column is populated.
        if len(tests) > 5_000_000:
            return

        text = tests[target].astype("object")
        missing = text.map(lambda value: str(value).strip() in {"", "<NA>", "nan", "None"})
        if bool(missing.any()):
            tests.loc[missing, target] = tests.loc[missing, source]

    # A present-but-empty test_id is not a usable identity. Keep this check
    # object-backed and short-circuit when the column is entirely non-null:
    # full LRTS runs can exceed 100M rows, and pandas may otherwise dispatch
    # `.str.strip()` to pyarrow, causing multi-GB temporary allocations.
    if "test_id" in tests.columns and "test_name" in tests.columns:
        _fill_missing_identity("test_id", "test_name")
    if "test_name" in tests.columns and "test_id" in tests.columns:
        _fill_missing_identity("test_name", "test_id")

    return tests


def _normalize_builds(builds: pd.DataFrame) -> pd.DataFrame:
    builds = _materialize_non_arrow(builds)

    builds = _stringify_existing(
        builds,
        [
            "build_id",
            "project",
            "status",
            "branch",
            "commit_sha",
        ],
    )

    if "build_started_at" in builds.columns:
        builds["build_started_at"] = pd.to_datetime(
            builds["build_started_at"],
            errors="coerce",
            utc=True,
        )

    builds = _dropna_existing(
        builds,
        [
            "build_id",
            "project",
            "build_started_at",
        ],
    )

    builds = _deduplicate_existing(
        builds,
        [
            "project",
            "build_id",
        ],
    )

    build_sort_cols = [
        col
        for col in [
            "project",
            "build_started_at",
            "build_id",
        ]
        if col in builds.columns
    ]

    if build_sort_cols:
        builds = builds.sort_values(
            build_sort_cols,
            kind="mergesort",
            ignore_index=True,
        )
    else:
        builds = builds.reset_index(drop=True)

    return builds


def _normalize_tests(tests: pd.DataFrame) -> pd.DataFrame:
    tests = _materialize_non_arrow(tests)

    tests = _ensure_test_identity_columns(tests)

    tests = _stringify_existing(
        tests,
        [
            "build_id",
            "project",
            "test_id",
            "test_name",
            "test_class",
            "test_suite",
            "file",
            "status",
            "test_status",
            "outcome",
        ],
    )

    tests = _ensure_test_identity_columns(tests)

    if "started_at" in tests.columns:
        tests["started_at"] = pd.to_datetime(
            tests["started_at"],
            errors="coerce",
            utc=True,
        )

    if "duration" not in tests.columns and "duration_sec" in tests.columns:
        tests["duration"] = tests["duration_sec"]

    if "duration" in tests.columns:
        tests["duration"] = tests["duration"].map(normalize_duration)

    failure_sources = []

    if "is_failed" in tests.columns:
        failure_sources.append(_to_bool(tests["is_failed"]))

    # Prefer raw status/verdict columns as a recovery path. Some dataset
    # loaders create an is_failed boolean early; if that early mapping is too
    # narrow, preserving only the boolean can silently erase failures. OR all
    # known status-like sources so a raw "timeout"/"regression"/"1" can still
    # become a failure label.
    for col in ["test_status", "outcome", "status"]:
        if col in tests.columns:
            failure_sources.append(_to_bool(tests[col]))

    if failure_sources:
        is_failed = failure_sources[0].copy()
        for source in failure_sources[1:]:
            is_failed = is_failed | source
        tests["is_failed"] = is_failed.fillna(False).astype(bool)
    else:
        tests["is_failed"] = False

    tests = _dropna_existing(
        tests,
        [
            "project",
            "test_name",
            "started_at",
        ],
    )

    tests = _deduplicate_existing(
        tests,
        [
            "project",
            "build_id",
            "test_id",
            "test_name",
            "started_at",
        ],
    )

    # Important:
    # Do not globally sort normalized_tests.csv.
    # Sorting the full test table caused large Arrow/Pandas memory allocations.
    tests = tests.reset_index(drop=True)

    return tests


def finalize_tables(
    builds: pd.DataFrame,
    tests: pd.DataFrame,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-safe normalization for large CI/CD datasets.

    Key design choices:
    - Avoids global sorting of the full tests table.
    - Avoids full-row drop_duplicates() on large test execution data.
    - Converts Arrow-backed columns before heavy pandas operations.
    - Preserves backward-compatible normalize_outcome() behavior.
    """

    before_build_rows = len(builds) if builds is not None else 0
    before_test_rows = len(tests) if tests is not None else 0

    builds = _normalize_builds(builds)
    tests = _normalize_tests(tests)

    after_build_rows = len(builds)
    after_test_rows = len(tests)

    if logger:
        logger.info(
            "Normalized builds=%d -> %d, tests=%d -> %d; dropped_build_rows=%d; dropped_test_rows=%d",
            before_build_rows,
            after_build_rows,
            before_test_rows,
            after_test_rows,
            before_build_rows - after_build_rows,
            before_test_rows - after_test_rows,
        )

    gc.collect()

    return builds, tests
