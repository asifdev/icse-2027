import logging
from pathlib import Path

def configure_logging(out: Path, verbose: bool = True) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ci_driftlab")
    log.handlers.clear(); log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(out / "run.log", encoding="utf-8"); fh.setFormatter(fmt); log.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    return log
