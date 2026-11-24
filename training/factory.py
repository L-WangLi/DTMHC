"""Load the final per-dataset settings, trajectories and DTMHC model."""

import json
import math
from pathlib import Path

from dataset.cmapss import load_cmapss
from dataset.common import add_condition_residuals
from dataset.ncmapss import load_ncmapss
from models import DTMHC

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("FD001", "FD002", "FD003", "FD004", "DS01", "DS03", "DS04",
            "DS05", "DS06", "DS07", "DS08a", "DS08c")


def load_config(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    family = "ncmapss" if dataset.startswith("DS") else "cmapss"
    cfg = json.loads((ROOT / "configs" / family / f"{dataset}.json").read_text())
    protocols = json.loads((ROOT / "configs/protocols.json").read_text())
    cfg["protocol"] = protocols["subsets"][dataset]
    return cfg


def engine_splits(protocol):
    """Expand inclusive ranges from the centralized protocol or checkpoint snapshot."""
    first, last = protocol["development_range"]
    validation = protocol["validation_units"]
    test_first, test_last = protocol["test_range"]
    return dict(train=[u for u in range(first, last + 1) if u not in validation],
                val=list(validation), test=list(range(test_first, test_last + 1)))


def build_model(cfg):
    return DTMHC(cfg["in_dim"], **cfg["model"])


def load_data(cfg, data_dir=None):
    """data_dir is a raw C-MAPSS folder or parent of processed N-CMAPSS subsets."""
    name = cfg["dataset"]
    if name.startswith("DS"):
        data = load_ncmapss(name, data_dir) if data_dir is not None else load_ncmapss(name)
    else:
        kwargs = dict(split_seed=cfg["split_seed"])
        if data_dir is not None:
            kwargs["raw_dir"] = Path(data_dir)
        data = load_cmapss(name, **kwargs)
    if len(data.feat_cols) != cfg["in_dim"] or data.window != cfg["window"]:
        raise ValueError("Feature dimension/window differs from the final configuration")
    expected_splits = engine_splits(cfg["protocol"])
    for split in ("train", "val", "test"):
        actual = [t.unit for t in getattr(data, split)]
        if actual != expected_splits[split]:
            raise ValueError(f"{name}: unexpected {split} engine split: {actual}")
    longest = max(len(t.rul) for t in data.train + data.val + data.test)
    expected = max(math.ceil(560 / cfg["hist_seg"]), math.ceil(longest / cfg["hist_seg"]) + 2)
    if expected != cfg["hist_slots"]:
        raise ValueError(f"History capacity {expected} differs from configured {cfg['hist_slots']}")
    if cfg["hist_condres"]:
        add_condition_residuals(data.train, data.val, data.test)
    return data
