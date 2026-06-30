from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from ci_driftlab.config import resolve_dataset_path
from ci_driftlab.data_loading import make_loader, loader_input_paths
from ci_driftlab.preprocessing.normalize import finalize_tables, normalize_outcome
from ci_driftlab.preprocessing.validation import (
    validate_required_columns,
    data_quality_report,
)
from ci_driftlab.windowing.temporal_windows import create_temporal_windows
from ci_driftlab.features.feature_extraction import extract_features
from ci_driftlab.tcp_baselines.baselines import evaluate_baselines
from ci_driftlab.io_utils import write_csv


def _project_shard_name(project: object) -> str:
    text = str(project)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.csv"


def run_phase1(config, dataset, out, logger, checkpoint=None):
    target = Path(out) / "phase1_foundation"
    target.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(config, dataset)

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

    def _remove_arrow_backing(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Defensive conversion.

        Some pandas installations read CSV columns as Arrow-backed arrays when
        pyarrow is installed. Large boolean filtering can then call Arrow
        take(), causing very large memory allocations.
        """

        for col in frame.columns:
            dtype_text = str(frame[col].dtype).lower()
            if "pyarrow" in dtype_text or dtype_text.endswith("[pyarrow]"):
                frame[col] = frame[col].astype("object")

        return frame

    def _read_csv_existing(path: Path, usecols=None) -> pd.DataFrame:
        path = Path(path)

        if usecols:
            wanted = set(usecols)
            frame = pd.read_csv(
                path,
                usecols=lambda c: c in wanted,
                low_memory=False,
            )
        else:
            frame = pd.read_csv(path, low_memory=False)

        return _remove_arrow_backing(frame)

    def _write_test_activity_index(tests: pd.DataFrame, path: Path) -> None:
        """
        Compact per-project timestamp counts for temporal windowing.

        Full LRTS normalized_tests.csv can be tens of GB. Windowing only needs
        counts by project/time, so keep a build/test-activity index instead of
        re-reading the full execution table.
        """

        required = {"project", "started_at"}
        if not required.issubset(tests.columns):
            write_csv(pd.DataFrame(columns=["project", "started_at", "test_count"]), path)
            return

        frame = tests.loc[:, ["project", "started_at"]].copy()
        frame["project"] = frame["project"].astype("string[python]")
        frame["started_at"] = pd.to_datetime(frame["started_at"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["project", "started_at"])

        if frame.empty:
            activity = pd.DataFrame(columns=["project", "started_at", "test_count"])
        else:
            activity = (
                frame.groupby(["project", "started_at"], observed=True)
                .size()
                .reset_index(name="test_count")
            )

        write_csv(activity, path)

    def _write_test_shards(tests: pd.DataFrame, shard_dir: Path) -> None:
        """
        Write project-scoped normalized test CSVs.

        Features and baselines can then process one project at a time instead
        of materializing the full LRTS test table.
        """

        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        manifest = {}

        if tests.empty or "project" not in tests.columns:
            (shard_dir / "manifest.json").write_text("{}", encoding="utf-8")
            return

        shard_columns = [
            c
            for c in [
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
            if c in tests.columns
        ]

        for project, group in tests.groupby("project", sort=False, observed=True):
            project_text = str(project)
            filename = _project_shard_name(project_text)
            group.loc[:, shard_columns].to_csv(shard_dir / filename, index=False)
            manifest[project_text] = {
                "file": filename,
                "rows": int(len(group)),
            }

        temp = shard_dir / "manifest.json.tmp"
        temp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(shard_dir / "manifest.json")

    def _read_test_shard(project: object, usecols_tests=None) -> pd.DataFrame:
        manifest_path = target / "test_shards" / "manifest.json"
        if not manifest_path.exists():
            return _read_csv_existing(target / "normalized_tests.csv", usecols=usecols_tests)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        item = manifest.get(str(project))
        if not item:
            return pd.DataFrame(columns=usecols_tests or [])

        return _read_csv_existing(
            target / "test_shards" / item["file"],
            usecols=usecols_tests,
        )

    def canonical(usecols_tests=None, usecols_builds=None):
        """
        Load normalized Phase 1 tables.

        usecols_tests and usecols_builds allow memory-heavy steps to load only
        the columns they actually need.
        """

        builds = _read_csv_existing(
            target / "normalized_builds.csv",
            usecols=usecols_builds,
        )

        tests = _read_csv_existing(
            target / "normalized_tests.csv",
            usecols=usecols_tests,
        )

        if "project" in builds.columns:
            builds["project"] = builds["project"].astype("string[python]")

        if "project" in tests.columns:
            tests["project"] = tests["project"].astype("string[python]")

        if "build_id" in builds.columns:
            builds["build_id"] = builds["build_id"].astype("string[python]")

        if "build_id" in tests.columns:
            tests["build_id"] = tests["build_id"].astype("string[python]")

        if "test_id" in tests.columns:
            tests["test_id"] = tests["test_id"].astype("string[python]")

        if "test_name" in tests.columns:
            tests["test_name"] = tests["test_name"].astype("string[python]")

        if "test_class" in tests.columns:
            tests["test_class"] = tests["test_class"].astype("string[python]")

        if "build_started_at" in builds.columns:
            builds["build_started_at"] = pd.to_datetime(
                builds["build_started_at"],
                errors="coerce",
                utc=True,
            )

        if "started_at" in tests.columns:
            tests["started_at"] = pd.to_datetime(
                tests["started_at"],
                errors="coerce",
                utc=True,
            )

        if "duration" in tests.columns:
            tests["duration"] = pd.to_numeric(tests["duration"], errors="coerce")

        if "is_failed" in tests.columns:
            tests["is_failed"] = tests["is_failed"].map(normalize_outcome).eq("fail")

        for col in ["test_status", "status", "outcome"]:
            if col in tests.columns:
                mapped = tests[col].map(normalize_outcome).eq("fail")
                if "is_failed" in tests.columns:
                    tests["is_failed"] = tests["is_failed"] | mapped
                else:
                    tests["is_failed"] = mapped

        return builds, tests

    def _write_failure_label_diagnostics(tests: pd.DataFrame, path: Path) -> None:
        """
        Persist the raw-status -> normalized-status -> is_failed audit trail.

        This is intentionally generated immediately after normalization, before
        windowing/features/baselines, so TCP failures can be debugged without
        rerunning every downstream phase.
        """

        rows = []
        source_cols = [c for c in ["test_status", "outcome", "status", "is_failed"] if c in tests.columns]

        if not source_cols:
            write_csv(
                pd.DataFrame(
                    columns=[
                        "source_column",
                        "raw_value",
                        "normalized_outcome",
                        "rows",
                        "mapped_failures",
                        "projects",
                    ]
                ),
                path,
            )
            return

        for col in source_cols:
            frame = tests.loc[:, [c for c in ["project", col] if c in tests.columns]].copy()
            frame["_raw_value"] = frame[col].astype("object").map(
                lambda value: "<NA>" if pd.isna(value) else str(value).strip()
            )
            frame["_normalized_outcome"] = frame[col].map(normalize_outcome)
            frame["_mapped_failure"] = frame["_normalized_outcome"].eq("fail")

            grouped = frame.groupby(
                ["_raw_value", "_normalized_outcome"],
                dropna=False,
                observed=True,
            )

            for (raw_value, normalized), group in grouped:
                rows.append(
                    {
                        "source_column": col,
                        "raw_value": raw_value,
                        "normalized_outcome": normalized,
                        "rows": int(len(group)),
                        "mapped_failures": int(group["_mapped_failure"].sum()),
                        "projects": int(group["project"].nunique())
                        if "project" in group.columns
                        else 0,
                    }
                )

        diagnostic = pd.DataFrame(rows)
        if not diagnostic.empty:
            diagnostic = diagnostic.sort_values(
                ["source_column", "rows", "raw_value"],
                ascending=[True, False, True],
                kind="mergesort",
            ).reset_index(drop=True)

        write_csv(diagnostic, path)

    def load_normalize():
        loader = make_loader(dataset, dataset_path, config, logger)

        builds, tests, schema = loader.load()
        builds, tests = finalize_tables(builds, tests, logger)

        validate_required_columns(
            builds,
            ["build_id", "project", "build_started_at"],
            "builds",
        )

        validate_required_columns(
            tests,
            ["build_id", "project", "test_id", "is_failed", "started_at"],
            "tests",
        )

        write_csv(builds, target / "normalized_builds.csv")
        write_csv(tests, target / "normalized_tests.csv")
        _write_failure_label_diagnostics(tests, target / "failure_label_diagnostics.csv")
        _write_test_activity_index(tests, target / "test_activity_index.csv")
        _write_test_shards(tests, target / "test_shards")
        write_csv(data_quality_report(builds, tests), target / "data_quality_report.csv")

        lines = ["# Schema mapping report", ""]

        for item in schema:
            lines.extend(
                [
                    f"## `{item['file']}`",
                    f"- Rows: {item.get('rows', 'unknown')}",
                    f"- Mapping: `{item.get('mapping', {})}`",
                    f"- Error: {item['error']}" if item.get("error") else "",
                ]
            )

        temp = target / "schema_mapping_report.md.tmp"
        temp.write_text("\n".join(lines), encoding="utf-8")
        temp.replace(target / "schema_mapping_report.md")

    raw_inputs = loader_input_paths(dataset, dataset_path, config)

    execute(
        "phase1.load_normalize",
        [
            target / "normalized_builds.csv",
            target / "normalized_tests.csv",
            target / "data_quality_report.csv",
            target / "failure_label_diagnostics.csv",
            target / "schema_mapping_report.md",
            target / "test_activity_index.csv",
            target / "test_shards",
        ],
        load_normalize,
        [*raw_inputs, config.source],
        {
            "dataset": dataset,
            "window_candidates": config["windowing"],
        },
    )

    def windowing():
        """
        Memory-safe windowing.

        Windowing only needs:
        - builds: project, build_id, build_started_at
        - tests: project, started_at
        """

        builds = _read_csv_existing(
            target / "normalized_builds.csv",
            usecols=[
                "project",
                "build_id",
                "build_started_at",
            ],
        )
        tests = _read_csv_existing(
            target / "test_activity_index.csv",
            usecols=["project", "started_at", "test_count"],
        )

        if "project" in builds.columns:
            builds["project"] = builds["project"].astype("string[python]")
        if "build_id" in builds.columns:
            builds["build_id"] = builds["build_id"].astype("string[python]")
        if "build_started_at" in builds.columns:
            builds["build_started_at"] = pd.to_datetime(
                builds["build_started_at"],
                errors="coerce",
                utc=True,
            )

        wc = config["windowing"]

        windows = create_temporal_windows(
            builds,
            tests,
            wc["granularities"],
            wc["strategies"],
            wc.get("min_builds_per_window", 1),
            wc.get("min_tests_per_window", 1),
        )

        if config["run"].get("mode") == "smoke" and not windows.empty:
            projects = windows.project.drop_duplicates().head(
                config["baseline_models"].get("smoke_limit_projects", 1)
            )

            windows = windows[windows.project.isin(projects)]

            windows = windows.groupby(
                ["project", "granularity", "strategy"],
                group_keys=False,
            ).head(config["baseline_models"].get("smoke_limit_windows", 3))

        write_csv(windows, target / "project_windows.csv")

    execute(
        "phase1.windowing",
        [target / "project_windows.csv"],
        windowing,
        [
            target / "normalized_builds.csv",
            target / "test_activity_index.csv",
        ],
        {
            "windowing": config["windowing"],
            "mode": config["run"].get("mode"),
        },
    )

    def features():
        """
        Memory-safe feature extraction.

        Feature extraction needs more test columns than windowing, but still
        does not need the full raw normalized_tests table.
        """

        builds = _read_csv_existing(
            target / "normalized_builds.csv",
            usecols=[
                "project",
                "build_id",
                "build_started_at",
                "status",
                "branch",
                "commit_sha",
            ],
        )

        windows = _read_csv_existing(target / "project_windows.csv")

        if "project" in builds.columns:
            builds["project"] = builds["project"].astype("string[python]")
        if "build_id" in builds.columns:
            builds["build_id"] = builds["build_id"].astype("string[python]")
        if "build_started_at" in builds.columns:
            builds["build_started_at"] = pd.to_datetime(
                builds["build_started_at"],
                errors="coerce",
                utc=True,
            )

        wf_parts = []
        tf_parts = []

        projects = windows["project"].dropna().astype(str).drop_duplicates().tolist()
        for index, project in enumerate(projects, start=1):
            logger.info("Feature extraction project %d/%d: %s", index, len(projects), project)
            project_windows = windows[windows["project"].astype(str).eq(project)]
            project_builds = builds[builds["project"].astype(str).eq(project)]
            project_tests = _read_test_shard(
                project,
                usecols_tests=[
                    "project",
                    "build_id",
                    "test_id",
                    "test_name",
                    "test_class",
                    "started_at",
                    "duration",
                    "is_failed",
                    "test_status",
                    "status",
                    "outcome",
                ],
            )
            wf_part, tf_part = extract_features(
                project_windows,
                project_builds,
                project_tests,
                dataset,
            )
            wf_parts.append(wf_part)
            tf_parts.append(tf_part)

        wf = pd.concat(wf_parts, ignore_index=True) if wf_parts else pd.DataFrame()
        tf = pd.concat(tf_parts, ignore_index=True) if tf_parts else pd.DataFrame()

        write_csv(wf, target / "features_by_window.csv")
        write_csv(tf, target / "test_features_by_window.csv")

    execute(
        "phase1.features",
        [
            target / "features_by_window.csv",
            target / "test_features_by_window.csv",
        ],
        features,
        [
            target / "project_windows.csv",
            target / "normalized_builds.csv",
            target / "test_shards",
        ],
        {
            "features": config["features"],
        },
    )

    def baselines():
        """
        Memory-safe TCP baseline evaluation.

        Baselines only need the test execution columns required to compute
        ranking history and APFD/APFDc/NAPFD.
        """

        windows = _read_csv_existing(target / "project_windows.csv")

        baseline_parts = []
        projects = windows["project"].dropna().astype(str).drop_duplicates().tolist()
        for index, project in enumerate(projects, start=1):
            logger.info("TCP baseline project %d/%d: %s", index, len(projects), project)
            project_windows = windows[windows["project"].astype(str).eq(project)]
            project_tests = _read_test_shard(
                project,
                usecols_tests=[
                    "project",
                    "build_id",
                    "test_id",
                    "test_name",
                    "test_class",
                    "started_at",
                    "duration",
                    "is_failed",
                    "test_status",
                    "status",
                    "outcome",
                ],
            )
            baseline_parts.append(
                evaluate_baselines(
                    project_windows,
                    project_tests,
                    dataset,
                    config["baseline_models"]["tcp"],
                    config.seed,
                    logger,
                )
            )

        baseline = (
            pd.concat(baseline_parts, ignore_index=True)
            if baseline_parts
            else pd.DataFrame()
        )

        write_csv(baseline, target / "baseline_tcp_results.csv")

    execute(
        "phase1.baselines",
        [target / "baseline_tcp_results.csv"],
        baselines,
        [
            target / "project_windows.csv",
            target / "test_shards",
        ],
        {
            "models": config["baseline_models"],
            "seed": config.seed,
        },
    )

    def summaries():
        baseline = _read_csv_existing(target / "baseline_tcp_results.csv")
        windows = _read_csv_existing(target / "project_windows.csv")

        if not baseline.empty:
            required_baseline_cols = [
                "granularity",
                "strategy",
                "model",
                "APFD",
                "APFDc",
                "NAPFD",
                "window_id",
            ]

            missing = [c for c in required_baseline_cols if c not in baseline.columns]
            if missing:
                raise ValueError(
                    f"baseline_tcp_results.csv missing required columns: {missing}"
                )

            protocol = baseline.groupby(
                ["granularity", "strategy", "model"],
                as_index=False,
            ).agg(
                APFD_mean=("APFD", "mean"),
                APFDc_mean=("APFDc", "mean"),
                NAPFD_mean=("NAPFD", "mean"),
                windows=("window_id", "nunique"),
            )
        else:
            protocol = pd.DataFrame(
                columns=[
                    "granularity",
                    "strategy",
                    "model",
                    "APFD_mean",
                    "APFDc_mean",
                    "NAPFD_mean",
                    "windows",
                ]
            )

        write_csv(protocol, target / "protocol_comparison.csv")

        if not windows.empty:
            required_window_cols = [
                "project",
                "granularity",
                "strategy",
                "window_id",
                "num_builds_train",
                "num_builds_test",
                "num_tests_train",
                "num_tests_test",
            ]

            missing = [c for c in required_window_cols if c not in windows.columns]
            if missing:
                raise ValueError(f"project_windows.csv missing required columns: {missing}")

            stats = windows.groupby(
                ["project", "granularity", "strategy"],
                as_index=False,
            ).agg(
                windows=("window_id", "count"),
                mean_train_builds=("num_builds_train", "mean"),
                mean_test_builds=("num_builds_test", "mean"),
                mean_train_tests=("num_tests_train", "mean"),
                mean_test_tests=("num_tests_test", "mean"),
            )
        else:
            stats = pd.DataFrame(
                columns=[
                    "project",
                    "granularity",
                    "strategy",
                    "windows",
                    "mean_train_builds",
                    "mean_test_builds",
                    "mean_train_tests",
                    "mean_test_tests",
                ]
            )

        write_csv(stats, target / "window_statistics.csv")

    execute(
        "phase1.summaries",
        [
            target / "protocol_comparison.csv",
            target / "window_statistics.csv",
        ],
        summaries,
        [
            target / "baseline_tcp_results.csv",
            target / "project_windows.csv",
        ],
    )

    logger.info("Phase 1 complete (checkpoint-aware)")
