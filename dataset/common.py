"""Trajectory containers, masked windows and train-fitted condition residuals."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

@dataclass
class Trajectory:
    unit: int
    x: np.ndarray
    sensors: np.ndarray
    rul: np.ndarray
    cycle: np.ndarray
    cond: np.ndarray


@dataclass
class SubsetData:
    name: str
    window: int
    feat_cols: List[str]
    sensor_cols: List[str]
    train: List[Trajectory]
    val: List[Trajectory]
    test: List[Trajectory]
    test_final_rul: np.ndarray
    info: Dict = field(default_factory=dict)
    test_rul_raw: np.ndarray = None


def train_window_index(trajs: List[Trajectory], window: int, short_prefixes: bool=False) -> np.ndarray:
    """[N, 2] (traj_idx, end_idx) for all stride-1 windows, in reference order.
    short_prefixes=True also adds windows ending at cycles 0..window-2 (left-padded by the
    protocol rule), so training covers the short histories met by short test units and by
    the multi-prefix evaluation. The reference protocol (frozen arrays) has full windows only."""
    idx = []
    for (k, tr) in enumerate(trajs):
        for end in range(0 if short_prefixes else window - 1, len(tr.rul)):
            idx.append((k, end))
    return np.asarray(idx, dtype=np.int64).reshape(-1, 2)


def gather_windows(trajs: List[Trajectory], index: np.ndarray, window: int, attr: str='x'):
    """Materialise windows ending at index[:,1]; left-pads with the first row when the
    history is shorter than `window` (reference make_padded_test_window rule).
    Returns X [N, W, D] float32 and pad mask [N, W] bool (True = padded)."""
    first = getattr(trajs[0], attr)
    X = np.empty((len(index), window, first.shape[1]), dtype=np.float32)
    pad = np.zeros((len(index), window), dtype=bool)
    for (n, (k, end)) in enumerate(index):
        arr = getattr(trajs[k], attr)
        start = end - window + 1
        if start >= 0:
            X[n] = arr[start:end + 1]
        else:
            n_pad = -start
            X[n, :n_pad] = arr[0]
            X[n, n_pad:] = arr[:end + 1]
            pad[n, :n_pad] = True
    return (X, pad)


def test_window_index(trajs: List[Trajectory]) -> np.ndarray:
    return np.asarray([(k, len(t.rul) - 1) for (k, t) in enumerate(trajs)], dtype=np.int64)


def add_condition_residuals(train: List[Trajectory], *groups: List[Trajectory]):
    """x_hist = x - (per-condition-cluster mean of x over TRAIN units). Used only for history
    statistics in the multi-condition C-MAPSS configuration."""
    X = np.concatenate([t.x for t in train])
    C = np.concatenate([t.cond for t in train])
    mu = {c: X[C == c].mean(0) for c in np.unique(C)}
    for g in (train,) + groups:
        for t in g:
            t.x_hist = (t.x - np.stack([mu[c] for c in t.cond])).astype(np.float32)
