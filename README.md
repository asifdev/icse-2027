# ICSE 2026 LRTS Result Reproduction Code

This archive contains the code needed to regenerate the CI-DriftLab experiment results for the LRTS dataset.

## Contents

- `src/ci_driftlab/`: result-generation package for LRTS loading, preprocessing, temporal windows, feature extraction, TCP baselines, drift detection, label/audit simulation, cost modeling, and retraining-policy summaries.
- `scripts/run_ci_drift_master.py`: main experiment runner.
- `scripts/test_dataset.py`: dataset preflight check.
- `scripts/inspect_lrts_schema.py`: schema inspection helper.
- `configs/master_lrts_ci_drift.yaml`: LRTS experiment configuration.
- `configs/datasets/lrts.yaml`: standalone LRTS dataset path reference.
- `pyproject.toml`: Python package metadata and dependencies.

## Requirements

- Python 3.10 or newer.
- LRTS raw dataset available locally.
- Python packages:

```powershell
python -m pip install -e ".[analysis]"
```

The `analysis` extra installs `scikit-learn`, `scipy`, `matplotlib`, and `pyarrow`, which are used by the full experiment path.

## LRTS Data Layout

Set the `datasets.lrts.path` value in `configs/master_lrts_ci_drift.yaml` to the local LRTS root. The expected layout is:

```text
<LRTS_ROOT>/
  dataset.csv
  processed_test_result/
    <project>/
      <pr_name>_build<build_id>/
        stage_<stage_id>/
          test_class.csv.zip
```

The bundled config currently points to the original local path used during development. Reviewers should replace it with their own LRTS extraction path before running.

## Preflight

From the archive root:

```powershell
python scripts/inspect_lrts_schema.py --config configs/master_lrts_ci_drift.yaml --dataset lrts --out schema_inspection_report.md
python scripts/test_dataset.py --config configs/master_lrts_ci_drift.yaml --dataset lrts --mode smoke --load --out results/lrts_preflight.json
```

## Smoke Run

Use smoke mode first to validate the environment and output contract quickly:

```powershell
python scripts/run_ci_drift_master.py --config configs/master_lrts_ci_drift.yaml --dataset lrts --phases 1,2,3,4 --mode smoke --out results/smoke_lrts
```

## Full Result Regeneration

Run the full LRTS experiment:

```powershell
python scripts/run_ci_drift_master.py --config configs/master_lrts_ci_drift.yaml --dataset lrts --phases 1,2,3,4 --mode full --out results/master_lrts_run_001
```

The main outputs are written under:

```text
results/master_lrts_run_001/
  run_manifest.json
  resolved_config.yaml
  summary_report.md
  checkpoints/pipeline_state.json
  phase1_foundation/
  phase2_drift_detection/
  phase3_label_cost_audit/
  phase4_recovery_retraining/
```

## Resume and Checkpoints

The runner is checkpoint-aware. If a run is interrupted, resume with:

```powershell
python scripts/run_ci_drift_master.py --config configs/master_lrts_ci_drift.yaml --dataset lrts --phases 1,2,3,4 --mode full --out results/master_lrts_run_001 --resume
```

To inspect checkpoint state without recomputing:

```powershell
python scripts/run_ci_drift_master.py --config configs/master_lrts_ci_drift.yaml --dataset lrts --out results/master_lrts_run_001 --checkpoint-status
```

To recompute from a specific step:

```powershell
python scripts/run_ci_drift_master.py --config configs/master_lrts_ci_drift.yaml --dataset lrts --phases 1,2,3,4 --mode full --out results/master_lrts_run_001 --restart-from phase2.evaluation
```

## Notes

- Raw LRTS data and generated result tables are not included.
- All reported result tables are computed from loaded LRTS data; missing or unreadable raw test archives are logged during loading.
