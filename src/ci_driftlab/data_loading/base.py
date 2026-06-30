from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class DatasetLoader(ABC):
    SUFFIXES = {".csv", ".parquet", ".json", ".jsonl", ".ndjson"}
    def __init__(self, root: Path, candidates: dict, logger, options=None):
        self.root, self.candidates, self.logger = Path(root), candidates, logger
        self.options = options or {}
    def discover(self):
        if not self.root.exists(): raise FileNotFoundError(f"Dataset folder does not exist: {self.root}. Place real data there or set datasets.<name>.path.")
        return sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in self.SUFFIXES)
    @abstractmethod
    def load(self): ...

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv": return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".parquet": return pd.read_parquet(path)
    try: return pd.read_json(path, lines=path.suffix.lower() in {".jsonl", ".ndjson"})
    except ValueError: return pd.json_normalize(pd.read_json(path).to_dict("records"))
