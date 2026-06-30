from pathlib import Path
from .lrts_loader import LRTSLoader

LOADERS={"lrts":LRTSLoader}

def make_loader(dataset,root,config,logger):
    if dataset not in LOADERS: raise ValueError(f"No loader registered for {dataset!r}; available: {sorted(LOADERS)}")
    options=dict(config["datasets"][dataset]); options["mode"]=config["run"].get("mode")
    options.setdefault("smoke_limit_projects",config["baseline_models"].get("smoke_limit_projects",1))
    return LOADERS[dataset](root,config["windowing"],logger,options)

def loader_input_paths(dataset,root,config):
    """Return exact raw inputs used for checkpoint invalidation."""
    root=Path(root); options=config["datasets"][dataset]; mode=config["run"].get("mode")
    if dataset=="lrts":
        paths=[root/"dataset.csv"]
        project=options.get("smoke_project") if mode=="smoke" else None
        paths.append(root/"processed_test_result"/project if project else root/"processed_test_result")
        return paths
    return [root]
