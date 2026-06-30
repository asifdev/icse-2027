import pandas as pd

def validate_required_columns(df, required, table):
    missing=[c for c in required if c not in df]
    if missing: raise ValueError(f"{table} missing required canonical columns: {missing}")

def data_quality_report(builds, tests):
    rows=[]
    for name,df in [("builds",builds),("tests",tests)]:
        large = len(df) > 5_000_000
        sample = df.head(100_000) if large else df
        for col in df.columns:
            if large:
                missing_count = None
                missing_rate = None
                unique_values = int(sample[col].nunique(dropna=True))
                unique_values_basis = f"first_{len(sample)}_rows"
            else:
                missing_count = int(df[col].isna().sum())
                missing_rate = float(df[col].isna().mean())
                unique_values = int(df[col].nunique(dropna=True))
                unique_values_basis = "exact"
            rows.append({"table":name,"column":col,"rows":len(df),"missing_count":missing_count,"missing_rate":missing_rate,"unique_values":unique_values,"unique_values_basis":unique_values_basis})
    return pd.DataFrame(rows)
