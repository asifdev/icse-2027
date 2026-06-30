#!/usr/bin/env python
from __future__ import annotations
import argparse,importlib.util,json,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from ci_driftlab.config import load_config
from ci_driftlab.logging_utils import configure_logging
from ci_driftlab.io_utils import write_json
from ci_driftlab.pipeline.phase1_foundation import run_phase1
from ci_driftlab.pipeline.phase2_drift_detection import run_phase2
from ci_driftlab.pipeline.phase3_label_cost_audit import run_phase3
from ci_driftlab.pipeline.phase4_recovery_retraining import run_phase4
from ci_driftlab.pipeline.checkpoints import CheckpointManager

RUNNERS={1:run_phase1,2:run_phase2,3:run_phase3,4:run_phase4}
ORDERED_STEPS=["phase1.load_normalize","phase1.windowing","phase1.features","phase1.baselines","phase1.summaries","phase2.generic_detectors","phase2.software_detectors","phase2.hybrid_detectors","phase2.evaluation","phase2.sensitivity","phase2.taxonomy","phase3.label_simulation","phase3.audit_simulation","phase3.cost_models","phase4.policy_simulation","phase4.summaries"]
def git_hash():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--dataset",default="lrts"); p.add_argument("--phases",default="1,2,3,4"); p.add_argument("--mode",choices=["smoke","full"],default=None); p.add_argument("--out",required=True); p.add_argument("--force",action="store_true",help="Recompute all requested steps"); p.add_argument("--resume",action="store_true",help="Reuse valid completed checkpoints and continue at the first unfinished step"); p.add_argument("--restart-from",help="Invalidate this step/phase and all later checkpoints, then resume"); p.add_argument("--checkpoint-status",action="store_true",help="Print checkpoint and artifact status without running"); a=p.parse_args()
    phases=sorted({int(x) for x in a.phases.split(",")}); invalid=set(phases)-set(RUNNERS)
    if invalid: p.error(f"Invalid phases: {sorted(invalid)}")
    out=Path(a.out).resolve()
    resumable=a.resume or bool(a.restart_from) or a.checkpoint_status
    if out.exists() and any(out.iterdir()) and not (a.force or resumable): raise SystemExit(f"Output folder is not empty: {out}. Use --resume to continue, --checkpoint-status to inspect, or --force to recompute.")
    out.mkdir(parents=True,exist_ok=True); cfg=load_config(a.config,a.dataset,phases,a.mode,a.out); log=configure_logging(out)
    checkpoints=CheckpointManager(out,log,resume=resumable,force=a.force)
    if a.restart_from: checkpoints.invalidate_from(a.restart_from,ORDERED_STEPS)
    if a.checkpoint_status:
        rows=checkpoints.status(ORDERED_STEPS); print("STEP\tSTATUS\tATTEMPT\tOUTPUTS_VALID\tELAPSED_SECONDS")
        for row in rows: print(f"{row['step']}\t{row['status']}\t{row['attempt']}\t{row['outputs_valid']}\t{row['elapsed_seconds'] or ''}")
        return 0
    import yaml
    temp=out/"resolved_config.yaml.tmp"; temp.write_text(yaml.safe_dump(cfg.data,sort_keys=False),encoding="utf-8"); temp.replace(out/"resolved_config.yaml")
    packages={x:importlib.util.find_spec(x) is not None for x in ["pandas","numpy","sklearn","scipy","yaml","matplotlib","pyarrow"]}
    manifest_path=out/"run_manifest.json"; prior=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and resumable else {}; invocations=prior.get("invocations",[]); invocations.append({"timestamp":datetime.now(timezone.utc).isoformat(),"phases":phases,"resume":resumable,"force":a.force,"restart_from":a.restart_from})
    manifest={"timestamp":prior.get("timestamp",datetime.now(timezone.utc).isoformat()),"last_invocation":invocations[-1]["timestamp"],"dataset":a.dataset,"mode":cfg["run"].get("mode"),"phases":phases,"config_path":str(cfg.source),"git_commit_hash":git_hash(),"python_version":platform.python_version(),"package_availability":packages,"output_folder":str(out),"checkpoint_state":str(checkpoints.path),"invocations":invocations}
    write_json(manifest,manifest_path); log.info("Starting dataset=%s phases=%s mode=%s resume=%s force=%s",a.dataset,phases,cfg["run"].get("mode"),resumable,a.force)
    try:
        for phase in phases: RUNNERS[phase](cfg,a.dataset,out,log,checkpoint=checkpoints)
    except KeyboardInterrupt:
        log.warning("Run interrupted safely. Resume with: python scripts/run_ci_drift_master.py --config %s --dataset %s --phases %s --mode %s --out %s --resume",a.config,a.dataset,a.phases,cfg["run"].get("mode"),out); return 130
    except Exception:
        log.error("Run stopped. Inspect checkpoints with --checkpoint-status, then rerun with --resume."); raise
    completed=sum(x["status"]=="completed" for x in checkpoints.status(ORDERED_STEPS)); summary=["# CI-DriftLab run summary","",f"- Dataset: `{a.dataset}`",f"- Mode: `{cfg['run'].get('mode')}`",f"- Phases completed: `{phases}`",f"- Valid completed checkpoints: `{completed}`",f"- Checkpoint state: `{checkpoints.path}`",f"- Output: `{out}`","","All outcome and duration values are computed from loaded data; dataset adapter limitations are recorded in the schema mapping report."]
    temp=out/"summary_report.md.tmp"; temp.write_text("\n".join(summary),encoding="utf-8"); temp.replace(out/"summary_report.md"); log.info("Run complete: %s",out); return 0
if __name__=="__main__": raise SystemExit(main())
