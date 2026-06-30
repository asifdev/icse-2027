_WARNED=set()
def ml_scores(train, test, model_name, logger):
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    except ImportError:
        if model_name not in _WARNED: logger.warning("scikit-learn unavailable; skipping %s for this run",model_name); _WARNED.add(model_name)
        return None
    features=["hist_failure","hist_duration"]
    x=train[features].fillna(0); y=train["is_failed"].astype(int)
    if len(x)<2 or y.nunique()<2: return None
    model=RandomForestClassifier(n_estimators=50,random_state=42) if model_name=="random_forest" else GradientBoostingClassifier(random_state=42)
    model.fit(x,y); return model.predict_proba(test[features].fillna(0))[:,1]
