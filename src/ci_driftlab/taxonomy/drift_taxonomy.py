import pandas as pd

def assign_taxonomy(generic,software):
    allr=pd.concat([generic,software],ignore_index=True)
    rows=[]
    mapping={"failure_rate":"failure_drift","recent_failure_rate":"failure_drift","duration_mean":"duration_drift","duration_std":"duration_drift","suite_size":"test_suite_drift","test_churn":"test_suite_drift","build_cadence":"build_behavior_drift","test_age":"project_activity_drift","APFDc":"model_performance_drift"}
    for (dataset,project,window),g in allr.groupby(["dataset","project","window_curr"]):
      cats=sorted({mapping.get(x) for x in g[g.drift_detected.astype(bool)].feature if mapping.get(x)})
      label="no_detected_drift" if not cats else cats[0] if len(cats)==1 else "mixed_drift"
      rows.append({"dataset":dataset,"project":project,"window_id":window,"taxonomy_label":label,"component_labels":";".join(cats),"num_triggered_signals":int(g.drift_detected.astype(bool).sum())})
    labels=pd.DataFrame(rows,columns=["dataset","project","window_id","taxonomy_label","component_labels","num_triggered_signals"])
    prevalence=labels.groupby("taxonomy_label",as_index=False).size().rename(columns={"size":"count"}) if not labels.empty else pd.DataFrame(columns=["taxonomy_label","count"])
    if not prevalence.empty: prevalence["prevalence"]=prevalence["count"]/prevalence["count"].sum()
    degradation=software[software.detector=="performance_degradation"][["dataset","project","window_curr","drift_detected","score"]].rename(columns={"window_curr":"window_id","drift_detected":"degradation_detected","score":"performance_drop"})
    relation=labels.merge(degradation,on=["dataset","project","window_id"],how="left")
    return labels,prevalence,relation
