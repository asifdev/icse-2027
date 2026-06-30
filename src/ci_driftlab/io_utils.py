from pathlib import Path
import json
import pandas as pd

def write_csv(df: pd.DataFrame, path: Path, columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        for col in columns:
            if col not in df: df[col] = pd.Series(dtype="object")
        df = df[list(columns)]
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Required prior-phase output missing: {path}")
    return pd.read_csv(path)

def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temp.replace(path)
