#!/usr/bin/env python
from __future__ import annotations
import argparse,logging,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from ci_driftlab.config import load_config,resolve_dataset_path
from ci_driftlab.data_loading import LRTSLoader
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(ROOT/"configs/master_lrts_ci_drift.yaml")); p.add_argument("--dataset",default="lrts"); p.add_argument("--path"); p.add_argument("--out",default="schema_inspection_report.md"); a=p.parse_args()
    cfg=load_config(a.config); path=Path(a.path).resolve() if a.path else resolve_dataset_path(cfg,a.dataset); log=logging.getLogger("schema")
    items=LRTSLoader(path,cfg["windowing"],log).inspect(); lines=["# LRTS schema inspection",f"",f"Dataset root: `{path}`",""]
    for x in items:
      print(x["file"]); print("  columns:",", ".join(x.get("columns",[]))); print("  mapping:",x.get("mapping"))
      lines.extend([f"## `{x['file']}`",f"- Rows: {x.get('rows','unknown')}",f"- Columns: `{', '.join(x.get('columns',[]))}`",f"- Likely mapping: `{x.get('mapping',{})}`",f"- Error: {x['error']}" if x.get("error") else ""])
    Path(a.out).write_text("\n".join(lines),encoding="utf-8"); print(f"Wrote {Path(a.out).resolve()}")
if __name__=="__main__": main()
