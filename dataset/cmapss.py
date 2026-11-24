"""C-MAPSS: split engines before fitting all scalers and condition clusters."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from .common import SubsetData, Trajectory

RAW_DIR = Path(__file__).resolve().parent / "raw/cmapss"
CONST_DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
VALID_SPLIT = 0.1
SPLIT_SEED = 42
N_CONDITION_CLUSTERS = 6
PROTOCOL = {'FD001': {'fd': 1, 'window': 40, 'include_ops': False, 'normalization': 'global_minmax'},
 'FD002': {'fd': 2,
           'window': 60,
           'include_ops': True,
           'normalization': 'condition_minmax'},
 'FD003': {'fd': 3, 'window': 40, 'include_ops': False, 'normalization': 'global_minmax'},
 'FD004': {'fd': 4,
           'window': 60,
           'include_ops': True,
           'normalization': 'condition_minmax'}}


def _read_txt(path: Path) -> pd.DataFrame:
    idx_cols = ['unit', 'cycle']
    op_cols  = [f'op{i}' for i in range(1, 4)]
    sensor_cols = [f's{i}' for i in range(1, 22)]
    cols = idx_cols + op_cols + sensor_cols
    return pd.read_csv(Path(path), sep=r'\s+', header=None, names=cols)


def add_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Capped training RUL: final cycle minus the current cycle."""
    df = df.copy()
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = (max_cycle - df['cycle']).clip(upper=cap)
    return df


def add_test_rul(test_df: pd.DataFrame, test_rul: np.ndarray,
                 cap: int = 125) -> pd.DataFrame:
    """Reconstruct capped per-cycle labels from official final test RUL."""
    df = test_df.copy()
    df['RUL'] = 0.0

    unit_order = list(df['unit'].drop_duplicates())
    if len(unit_order) != len(test_rul):
        raise ValueError(
            f"Engine count {len(unit_order)} differs from label count {len(test_rul)}")

    uid2idx = {u: i for i, u in enumerate(unit_order)}
    for uid, grp in df.groupby('unit'):
        final_rul = min(test_rul[uid2idx[uid]], cap)
        max_cyc   = grp['cycle'].max()
        for idx in grp.index:
            cur_cyc = df.loc[idx, 'cycle']
            df.loc[idx, 'RUL'] = min(final_rul + (max_cyc - cur_cyc), cap)
    return df


def global_minmax_scaling(train_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          feat_cols: List[str]
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Fit MinMax scaling on training engines and transform both groups."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feat_cols].values.astype(float))

    train_s = train_df.copy()
    test_s  = test_df.copy()
    train_s[feat_cols] = scaler.transform(train_df[feat_cols].values.astype(float))
    test_s[feat_cols]  = scaler.transform(test_df[feat_cols].values.astype(float))

    info = {
        'method':     'global_minmax',
        'feature_range': [0, 1],
        'data_min':   scaler.data_min_.tolist(),
        'data_max':   scaler.data_max_.tolist(),
        'scale':      scaler.scale_.tolist(),
        'data_range': scaler.data_range_.tolist(),
    }
    return train_s, test_s, info


def condition_wise_minmax_scaling(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    op_cols: List[str],
    sensor_cols: List[str],
    n_clusters: int = 6,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Condition-wise MinMax scaling for multi-operating-condition subsets.

    Clusters are fit on training op1/op2/op3 only. Sensors are then scaled by
    the MinMax scaler fitted on training rows within each operating-condition
    cluster. Test rows are assigned to the nearest training condition cluster
    without using labels.

    Ops are still globally MinMax-scaled so they remain available as model
    inputs while sensor normalization is condition-aware.
    """
    if len(op_cols) != 3:
        raise ValueError("condition_wise_minmax_scaling requires op1/op2/op3.")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")

    train_s = train_df.copy()
    test_s = test_df.copy()
    train_s[op_cols + sensor_cols] = train_s[op_cols + sensor_cols].astype(float)
    test_s[op_cols + sensor_cols] = test_s[op_cols + sensor_cols].astype(float)

    op_scaler = MinMaxScaler(feature_range=(0, 1))
    op_scaler.fit(train_df[op_cols].values.astype(float))
    train_ops_scaled = op_scaler.transform(train_df[op_cols].values.astype(float))
    test_ops_scaled = op_scaler.transform(test_df[op_cols].values.astype(float))

    n_unique_ops = np.unique(train_ops_scaled, axis=0).shape[0]
    k = min(int(n_clusters), int(n_unique_ops), len(train_df))
    if k < 1:
        raise ValueError("No training rows available for condition-wise normalization.")

    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    train_cluster = kmeans.fit_predict(train_ops_scaled)
    test_cluster = kmeans.predict(test_ops_scaled)

    train_s[op_cols] = train_ops_scaled
    test_s[op_cols] = test_ops_scaled

    cluster_info = []
    for cid in range(k):
        train_mask = train_cluster == cid
        test_mask = test_cluster == cid
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_df.loc[train_mask, sensor_cols].values.astype(float))
        train_s.loc[train_mask, sensor_cols] = scaler.transform(
            train_df.loc[train_mask, sensor_cols].values.astype(float)
        )
        if np.any(test_mask):
            test_s.loc[test_mask, sensor_cols] = scaler.transform(
                test_df.loc[test_mask, sensor_cols].values.astype(float)
            )
        cluster_info.append({
            'cluster': int(cid),
            'train_rows': int(train_mask.sum()),
            'test_rows': int(test_mask.sum()),
            'center': kmeans.cluster_centers_[cid].tolist(),
            'sensor_data_min': scaler.data_min_.tolist(),
            'sensor_data_max': scaler.data_max_.tolist(),
        })

    train_s['condition_cluster'] = train_cluster.astype(int)
    test_s['condition_cluster'] = test_cluster.astype(int)

    info = {
        'method': 'condition_wise_minmax',
        'feature_range': [0, 1],
        'op_cols': op_cols,
        'sensor_cols': sensor_cols,
        'n_requested_clusters': int(n_clusters),
        'n_used_clusters': int(k),
        'cluster_fit_data': 'train_op1_op2_op3_only',
        'test_assignment': 'nearest_training_op_cluster',
        'op_scaler': {
            'data_min': op_scaler.data_min_.tolist(),
            'data_max': op_scaler.data_max_.tolist(),
            'scale': op_scaler.scale_.tolist(),
            'data_range': op_scaler.data_range_.tolist(),
        },
        'clusters': cluster_info,
    }
    return train_s, test_s, info


def _split_train_units(
    train_df: pd.DataFrame,
    valid_split: float,
    random_seed: int,
) -> Tuple[List[int], List[int]]:
    """Decide engine-disjoint train/validation units before preprocessing fit."""
    units = np.array(sorted(int(u) for u in train_df['unit'].unique()), dtype=int)
    if valid_split <= 0:
        return units.tolist(), []
    rng = np.random.RandomState(random_seed)
    perm_units = rng.permutation(units)
    n_val_units = int(round(len(perm_units) * valid_split))
    n_val_units = min(max(n_val_units, 1), len(perm_units) - 1)
    val_units = sorted(int(u) for u in perm_units[:n_val_units])
    train_units = sorted(int(u) for u in perm_units[n_val_units:])
    return train_units, val_units


def load_raw(fd, raw_dir):
    raw_dir = Path(raw_dir)
    return (_read_txt(raw_dir / f"train_FD00{fd}.txt"),
            _read_txt(raw_dir / f"test_FD00{fd}.txt"),
            np.loadtxt(raw_dir / f"RUL_FD00{fd}.txt"))


def _to_trajs(df: pd.DataFrame, feat_cols, sensor_cols) -> List[Trajectory]:
    out = []
    has_cond = "condition_cluster" in df.columns
    for uid, g in df.groupby("unit"):
        out.append(Trajectory(
            unit=int(uid),
            x=g[feat_cols].values.astype(np.float32),
            sensors=g[sensor_cols].values.astype(np.float32),
            rul=g["RUL"].values.astype(np.float32),
            cycle=g["cycle"].values.astype(np.int32),
            cond=(g["condition_cluster"].values.astype(np.int16) if has_cond
                  else np.zeros(len(g), dtype=np.int16)),
        ))
    return out


def load_cmapss(name: str, raw_dir: Path = RAW_DIR, split_seed: int = SPLIT_SEED) -> SubsetData:
    """Load complete trajectories; split engines before fitting preprocessing.

    The condition-clustering seed stays 42. split_seed controls only the
    training/validation engine split.
    """
    p = PROTOCOL[name]
    train_raw, test_raw, test_rul = load_raw(p["fd"], raw_dir)
    drop = [c for c in CONST_DROP if c in train_raw.columns]
    train_clean = train_raw.drop(columns=drop)
    test_clean = test_raw.drop(columns=drop)
    sensor_cols = [c for c in train_clean.columns if c.startswith("s")]
    assert len(sensor_cols) == 14
    op_cols = [f"op{i}" for i in range(1, 4)] if p["include_ops"] else []
    feat_cols = op_cols + sensor_cols

    train_rul = add_rul(train_clean, cap=125)
    test_rul_df = add_test_rul(test_clean, test_rul, cap=125)

    tr_units, va_units = _split_train_units(train_rul, VALID_SPLIT, split_seed)
    fit_raw = train_rul[train_rul["unit"].isin(tr_units)].copy()
    val_raw = train_rul[train_rul["unit"].isin(va_units)].copy()
    eval_raw = pd.concat([val_raw.assign(_split="val"), test_rul_df.assign(_split="test")],
                         axis=0, ignore_index=True)
    if p["normalization"] == "condition_minmax":
        tr_s, ev_s, sc_info = condition_wise_minmax_scaling(
            fit_raw, eval_raw, op_cols, sensor_cols,
            n_clusters=N_CONDITION_CLUSTERS, random_seed=SPLIT_SEED)
    else:
        tr_s, ev_s, sc_info = global_minmax_scaling(fit_raw, eval_raw, feat_cols)
    va_s = ev_s[ev_s["_split"] == "val"].drop(columns=["_split"])
    te_s = ev_s[ev_s["_split"] == "test"].drop(columns=["_split"])

    test_trajs = _to_trajs(te_s, feat_cols, sensor_cols)
    return SubsetData(
        name=name, window=p["window"], feat_cols=feat_cols, sensor_cols=sensor_cols,
        train=_to_trajs(tr_s, feat_cols, sensor_cols),
        val=_to_trajs(va_s, feat_cols, sensor_cols),
        test=test_trajs,
        test_final_rul=np.array([t.rul[-1] for t in test_trajs], dtype=np.float32),
        test_rul_raw=np.asarray(test_rul, dtype=np.float32),
        info={"train_units": tr_units, "val_units": va_units,
              "scaler_fit_units": tr_units, "normalization": p["normalization"],
              "n_condition_clusters": sc_info.get("n_used_clusters", 1), "split_seed": split_seed},
    )
