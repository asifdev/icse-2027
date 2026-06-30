from pathlib import Path
import pandas as pd
from ci_driftlab.io_utils import read_csv,write_csv
from ci_driftlab.retraining_policies.retraining import simulate_retraining

def run_phase4(config,dataset,out,logger,checkpoint=None):
    root=Path(out); target=root/"phase4_recovery_retraining"; target.mkdir(parents=True,exist_ok=True)
    def execute(name,outputs,fn,inputs=(),parameters=None): return checkpoint.run_step(name,list(outputs),fn,list(inputs),parameters or {}) if checkpoint else fn()
    baseline_path=root/"phase1_foundation"/"baseline_tcp_results.csv"; software_path=root/"phase2_drift_detection"/"software_aware_detector_results.csv"; audit_path=root/"phase3_label_cost_audit"/"audit_strategy_results.csv"; policy_path=target/"retraining_policy_results.csv"
    execute("phase4.policy_simulation",[policy_path],lambda:write_csv(simulate_retraining(read_csv(baseline_path),read_csv(software_path),read_csv(audit_path),config["retraining_policies"],config["cost_model"]),policy_path),[baseline_path,software_path,audit_path],{"policies":config["retraining_policies"],"cost_model":config["cost_model"]})
    summary_outputs=[target/"recovery_time_by_project.csv",target/"policy_sensitivity.csv",target/"policy_ablation.csv",target/"unnecessary_retraining_rate.csv",target/"recovery_cost_benefit.csv"]
    def summaries():
        results=read_csv(policy_path); recovery=results.groupby(["dataset","project","policy"],as_index=False).agg(mean_recovery_time=("recovery_time","mean"),retraining_count=("retraining_count","sum"),missed_recovery_rate=("missed_recovery_rate","mean")) if not results.empty else pd.DataFrame(columns=["dataset","project","policy","mean_recovery_time","retraining_count","missed_recovery_rate"]); write_csv(recovery,summary_outputs[0])
        sens=[]
        for mult in config["cost_model"]["sensitivity_multipliers"]:
          for p,g in results.groupby("policy"): sens.append({"policy":p,"cost_multiplier":mult,"mean_total_cost":g.total_cost.mean()*mult,"mean_APFDc":g.APFDc.mean()})
        write_csv(pd.DataFrame(sens,columns=["policy","cost_multiplier","mean_total_cost","mean_APFDc"]),summary_outputs[1]); write_csv(results.groupby("policy",as_index=False).agg(mean_APFD=("APFD","mean"),mean_APFDc=("APFDc","mean"),mean_NAPFD=("NAPFD","mean"),total_retrainings=("retraining_count","sum"),mean_total_cost=("total_cost","mean")) if not results.empty else pd.DataFrame(columns=["policy","mean_APFD","mean_APFDc","mean_NAPFD","total_retrainings","mean_total_cost"]),summary_outputs[2]); write_csv(results.groupby("policy",as_index=False).agg(unnecessary_retraining_rate=("unnecessary_retraining_rate","mean"),retraining_count=("retraining_count","sum")) if not results.empty else pd.DataFrame(columns=["policy","unnecessary_retraining_rate","retraining_count"]),summary_outputs[3]); write_csv(results.groupby("policy",as_index=False).agg(mean_cost=("total_cost","mean"),mean_cost_benefit_score=("cost_benefit_score","mean"),mean_APFDc=("APFDc","mean"),missed_recovery_rate=("missed_recovery_rate","mean")) if not results.empty else pd.DataFrame(columns=["policy","mean_cost","mean_cost_benefit_score","mean_APFDc","missed_recovery_rate"]),summary_outputs[4])
    execute("phase4.summaries",summary_outputs,summaries,[policy_path],{"multipliers":config["cost_model"]["sensitivity_multipliers"]}); logger.info("Phase 4 complete (checkpoint-aware)")
