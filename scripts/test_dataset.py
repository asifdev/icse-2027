from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from ci_driftlab.config import load_config,resolve_dataset_path
from ci_driftlab.data_loading import make_loader
from ci_driftlab.preprocessing.normalize import finalize_tables


def main():
    parser=argparse.ArgumentParser(description="Preflight one configured dataset without running experiment phases")
    parser.add_argument("--config",default=str(ROOT/"configs/master_lrts_ci_drift.yaml"))
    parser.add_argument("--dataset",required=True,choices=["lrts"])
    parser.add_argument("--mode",choices=["smoke","full"],default="smoke")
    parser.add_argument("--load",action="store_true",help="Load and normalize data, not just inspect schema")
    parser.add_argument("--out",help="Optional JSON report path")
    args=parser.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s"); log=logging.getLogger("dataset_preflight")
    cfg=load_config(args.config,args.dataset,mode=args.mode); path=resolve_dataset_path(cfg,args.dataset); loader=make_loader(args.dataset,path,cfg,log); inspection=loader.inspect()
    report={"dataset":args.dataset,"mode":args.mode,"path":str(path),"exists":path.exists(),"inspection":inspection,"load_test":"not_requested"}
    if args.load:
        builds,tests,_=loader.load(); builds,tests=finalize_tables(builds,tests,log)
        report.update({"load_test":"passed","build_rows":len(builds),"test_rows":len(tests),"projects":sorted(builds.project.dropna().astype(str).unique().tolist()),"date_min":str(builds.build_started_at.min()),"date_max":str(builds.build_started_at.max()),"failed_test_rows":int(tests.is_failed.sum()),"missing_test_timestamps":int(tests.started_at.isna().sum())})
    rendered=json.dumps(report,indent=2,default=str)
    if args.out:
        target=Path(args.out); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(rendered,encoding="utf-8")
    print(rendered)


if __name__=="__main__": main()
