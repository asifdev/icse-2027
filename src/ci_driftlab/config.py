from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

REQUIRED = ("run", "datasets", "windowing", "features", "baseline_models", "drift_detectors", "label_budget", "cost_model", "retraining_policies", "paper_exports")

@dataclass
class ExperimentConfig:
    data: dict[str, Any]
    source: Path
    @property
    def seed(self) -> int: return int(self.data["run"].get("seed", 42))
    def __getitem__(self, key: str) -> Any: return self.data[key]

def load_config(path: str | Path, dataset: str | None = None, phases=None, mode: str | None = None, out: str | None = None) -> ExperimentConfig:
    source = Path(path).resolve()
    if not source.exists(): raise FileNotFoundError(f"Configuration not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    missing = [x for x in REQUIRED if x not in data]
    if missing: raise ValueError(f"Config missing required sections: {', '.join(missing)}")
    data = copy.deepcopy(data)
    data["run"].update({k:v for k,v in {"dataset":dataset, "mode":mode, "out":out}.items() if v is not None})
    if phases is not None: data["run"]["phases"] = list(phases)
    return ExperimentConfig(data, source)

def resolve_dataset_path(config: ExperimentConfig, dataset: str) -> Path:
    if dataset not in config["datasets"]: raise ValueError(f"Unknown dataset {dataset!r}; configured: {list(config['datasets'])}")
    raw = Path(config["datasets"][dataset]["path"])
    return raw if raw.is_absolute() else (config.source.parent.parent / raw).resolve()
