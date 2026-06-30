import pandas as pd

def simulate_audits(label_results,hybrid):
    base=label_results.sort_values(["budget_percent","delay"]).groupby(["dataset","project","window_id","sampling_policy"],as_index=False).first()
    audit=base.rename(columns={"sampling_policy":"audit_policy","available_labels":"audited_items","detected_degradation":"audit_signal"})
    audit["audit_yield"]=audit["audit_signal"].astype(int)/audit["audited_items"].clip(lower=1); audit["audit_cost_proxy"]=audit["audited_items"]
    audit=audit[["dataset","project","window_id","audit_policy","audited_items","audit_signal","audit_yield","audit_cost_proxy","missed_degradation"]]
    detector=hybrid[hybrid.rule=="weighted_score"][["dataset","project","window_curr","drift_detected","score"]].rename(columns={"window_curr":"window_id","drift_detected":"detector_signal"})
    merged=audit.merge(detector,on=["dataset","project","window_id"],how="left").fillna({"detector_signal":False,"score":0})
    rows=[]
    for r in merged.itertuples(index=False):
      for method,value in [("detector_only",r.detector_signal),("audit_only",r.audit_signal),("detector_or_audit",r.detector_signal or r.audit_signal),("detector_and_audit",r.detector_signal and r.audit_signal),("weighted_fusion",.5*float(r.detector_signal)+.5*float(r.audit_signal)>=.5)]:
        rows.append({"dataset":r.dataset,"project":r.project,"window_id":r.window_id,"audit_policy":r.audit_policy,"fusion_method":method,"detector_signal":bool(r.detector_signal),"audit_signal":bool(r.audit_signal),"combined_signal":bool(value)})
    fusion=pd.DataFrame(rows,columns=["dataset","project","window_id","audit_policy","fusion_method","detector_signal","audit_signal","combined_signal"])
    micro=audit.groupby(["audit_policy"],as_index=False).agg(windows=("window_id","count"),audited_items=("audited_items","sum"),detected_events=("audit_signal","sum"),mean_audit_yield=("audit_yield","mean"),audit_cost_proxy=("audit_cost_proxy","sum"))
    return audit,fusion,micro
