import pandas as pd

def compute_costs(evaluation,label_results,cfg):
    fa=[]; missed=[]
    for r in evaluation.itertuples(index=False):
      fa.append({"detector_family":r.detector_family,"detector":r.detector,"false_alarms":r.false_positive,"false_alarm_cost":r.false_positive*cfg["false_alarm_cost"],"unnecessary_retraining_cost":r.false_positive*cfg["retraining_cost"]})
      missed.append({"detector_family":r.detector_family,"detector":r.detector,"missed_drifts":r.false_negative,"missed_drift_cost":r.false_negative*cfg["missed_drift_cost"],"delayed_failure_cost":r.false_negative*cfg["delayed_failure_cost"]})
    fa=pd.DataFrame(fa,columns=["detector_family","detector","false_alarms","false_alarm_cost","unnecessary_retraining_cost"]); missed=pd.DataFrame(missed,columns=["detector_family","detector","missed_drifts","missed_drift_cost","delayed_failure_cost"])
    sens=[]
    for mult in cfg["sensitivity_multipliers"]:
      for r in evaluation.itertuples(index=False):
        label=label_results.label_cost_proxy.mean() if not label_results.empty else 0
        sens.append({"detector_family":r.detector_family,"detector":r.detector,"multiplier":mult,"false_alarm_cost":r.false_positive*cfg["false_alarm_cost"]*mult,"missed_drift_cost":r.false_negative*cfg["missed_drift_cost"]*mult,"label_cost_proxy":label,"total_cost":(r.false_positive*cfg["false_alarm_cost"]+r.false_negative*cfg["missed_drift_cost"])*mult+label})
    return fa,missed,pd.DataFrame(sens,columns=["detector_family","detector","multiplier","false_alarm_cost","missed_drift_cost","label_cost_proxy","total_cost"])
