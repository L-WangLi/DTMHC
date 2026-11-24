"""Reconstruct complete N-CMAPSS trajectories from the final frozen-window protocol."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .common import SubsetData, Trajectory, gather_windows

NCMAPSS_WINDOW = 60
NCMAPSS_DIR = Path(__file__).resolve().parent / "processed/ncmapss"

def _reconstruct(X, uid, y, fc, split_units):
    trajs = []
    for u in split_units:
        k = np.where(uid == u)[0]
        Xu, yu = X[k], y[k]
        seq = np.concatenate([Xu[0], Xu[1:, -1]], 0).astype(np.float32)        # [T, 18]
        T = len(seq)
        life_tokens_rul = np.empty(T, dtype=np.float32)
        life_tokens_rul[NCMAPSS_WINDOW - 1:] = yu
        cyc = np.arange(T) // 3 + 1                                              # flight cycle of token
        # tokens before the first full window: RUL from the cycle index (y + cycle is constant per unit)
        total = yu[0] + cyc[NCMAPSS_WINDOW - 1]
        life_tokens_rul[:NCMAPSS_WINDOW - 1] = total - cyc[:NCMAPSS_WINDOW - 1]
        trajs.append(Trajectory(unit=int(u), x=seq, sensors=seq[:, 4:], rul=life_tokens_rul,
                                cycle=cyc.astype(np.int32),
                                cond=np.full(T, int(fc[k[0]]), dtype=np.int16)))
    return trajs


def load_ncmapss(name: str, data_dir: Path = NCMAPSS_DIR) -> SubsetData:
    root = Path(data_dir) / name
    meta = json.load(open(root / "metadata.json"))
    out = {}
    for s in ("train", "val", "test"):
        X, uid, y = (np.load(root / f"{a}_{s}.npy") for a in ("X", "uid", "y"))
        fc = np.load(root / f"fc_{s}_window.npy")
        units = meta[f"{s}_units"]
        out[s] = _reconstruct(X, uid, y, fc, units)
        # exact reconstruction check against the frozen windows
        idx = np.array([(i, e) for i, t in enumerate(out[s]) for e in range(NCMAPSS_WINDOW - 1, len(t.rul))])
        Xr, _ = gather_windows(out[s], idx, NCMAPSS_WINDOW)
        order = np.concatenate([np.where(uid == u)[0] for u in units])
        assert np.array_equal(Xr, X[order].astype(np.float32)), f"{name}/{s}: reconstruction mismatch"
        assert np.allclose([out[s][i].rul[e] for i, e in idx], y[order]), f"{name}/{s}: label mismatch"
    return SubsetData(name=name, window=NCMAPSS_WINDOW, feat_cols=[f"f{i}" for i in range(18)],
                      sensor_cols=[f"s{i}" for i in range(14)], train=out["train"], val=out["val"],
                      test=out["test"], test_final_rul=np.array([t.rul[-1] for t in out["test"]]),
                      info={"train_units": meta["train_units"], "val_units": meta["val_units"],
                            "scaler_fit_units": meta["train_units"], "protocol": meta["protocol"],
                            "dataset": "N-CMAPSS"},
                      test_rul_raw=np.array([t.rul[-1] for t in out["test"]], dtype=np.float32))
