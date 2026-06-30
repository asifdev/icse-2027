from pathlib import Path
from ci_driftlab.io_utils import read_csv,write_csv
from ci_driftlab.label_budget.label_simulation import simulate_labels
from ci_driftlab.audit_policies.audit_simulation import simulate_audits
from ci_driftlab.cost_models.drift_costs import compute_costs

def run_phase3(config,dataset,out,logger,checkpoint=None):
    root=Path(out); target=root/"phase3_label_cost_audit"; target.mkdir(parents=True,exist_ok=True)
    policies=config["audit_policies"]; lb=config["label_budget"]
    def execute(name,outputs,fn,inputs=(),parameters=None): return checkpoint.run_step(name,list(outputs),fn,list(inputs),parameters or {}) if checkpoint else fn()
    tfp=root/"phase1_foundation"/"test_features_by_window.csv"; swp=root/"phase2_drift_detection"/"software_aware_detector_results.csv"; hyp=root/"phase2_drift_detection"/"hybrid_detector_results.csv"; evp=root/"phase2_drift_detection"/"detector_precision_recall_delay.csv"
    budget_path=target/"label_budget_results.csv"; delay_path=target/"label_delay_results.csv"
    def labels_step():
        software=read_csv(swp); budget,delays=simulate_labels(read_csv(tfp),software[software.detector=="performance_degradation"],lb["budgets_percent"],lb["delays"],policies,config.seed); write_csv(budget,budget_path); write_csv(delays,delay_path)
    execute("phase3.label_simulation",[budget_path,delay_path],labels_step,[tfp,swp],{"label_budget":lb,"policies":policies,"seed":config.seed})
    audit_outputs=[target/"audit_strategy_results.csv",target/"detector_audit_fusion.csv",target/"micro_audit_results.csv"]
    def audits_step():
        audit,fusion,micro=simulate_audits(read_csv(delay_path),read_csv(hyp)); write_csv(audit,audit_outputs[0]); write_csv(fusion,audit_outputs[1]); write_csv(micro,audit_outputs[2])
    execute("phase3.audit_simulation",audit_outputs,audits_step,[delay_path,hyp],{"policies":policies})
    cost_outputs=[target/"false_alarm_cost.csv",target/"missed_drift_cost.csv",target/"cost_sensitivity.csv"]
    def costs_step():
        fa,missed,sensitivity=compute_costs(read_csv(evp),read_csv(budget_path),config["cost_model"]); write_csv(fa,cost_outputs[0]); write_csv(missed,cost_outputs[1]); write_csv(sensitivity,cost_outputs[2])
    execute("phase3.cost_models",cost_outputs,costs_step,[evp,budget_path],{"cost_model":config["cost_model"]}); logger.info("Phase 3 complete (checkpoint-aware)")
