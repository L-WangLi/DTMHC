"""Final N-CMAPSS preprocessing: three equal-sample intervals per flight."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import h5py
import numpy as np

DATASETS = {'DS01': 'N-CMAPSS_DS01-005.h5',
 'DS03': 'N-CMAPSS_DS03-012.h5',
 'DS04': 'N-CMAPSS_DS04.h5',
 'DS05': 'N-CMAPSS_DS05.h5',
 'DS06': 'N-CMAPSS_DS06.h5',
 'DS07': 'N-CMAPSS_DS07.h5',
 'DS08a': 'N-CMAPSS_DS08a-009.h5',
 'DS08c': 'N-CMAPSS_DS08c-008.h5'}
N_OPS = 4
N_SENSORS = 14
INPUT_DIM = 18


def decode_names(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values.tolist()]


def load_raw(h5_path: Path) -> Dict:
    with h5py.File(h5_path, "r") as f:
        return {
            "A_dev": f["A_dev"][:],
            "W_dev": f["W_dev"][:],
            "X_s_dev": f["X_s_dev"][:],
            "Y_dev": f["Y_dev"][:, 0],
            "A_test": f["A_test"][:],
            "W_test": f["W_test"][:],
            "X_s_test": f["X_s_test"][:],
            "Y_test": f["Y_test"][:, 0],
            "A_var": decode_names(f["A_var"][:]),
            "W_var": decode_names(f["W_var"][:]),
            "X_s_var": decode_names(f["X_s_var"][:]),
        }


def aggregate_phase_tokens(
    A: np.ndarray,
    W: np.ndarray,
    Xs: np.ndarray,
    Y: np.ndarray,
    n_phase_tokens: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Aggregate each flight cycle into fixed phase tokens.

    N-CMAPSS flights have variable duration. Splitting each flight into equal
    relative-time bins gives each flight the same number of tokens and preserves
    relative within-flight order without identifying physical flight phases.
    """
    result: Dict[int, Dict[str, list]] = {}
    for uid in np.unique(A[:, 0]).astype(int):
        umask = A[:, 0] == uid
        fc = int(np.unique(A[umask, 2])[0])
        cycles = np.unique(A[umask, 1]).astype(int)
        rows = {"fc": fc, "W": [], "Xs": [], "Y": [], "cycle": [], "phase": [], "n_raw": []}
        for cyc in cycles:
            idx = np.flatnonzero(umask & (A[:, 1] == cyc))
            if idx.size == 0:
                continue
            splits = np.array_split(idx, n_phase_tokens)
            y_val = float(Y[idx[-1]])
            for phase_id, part in enumerate(splits):
                if part.size == 0:
                    continue
                rows["W"].append(W[part].mean(axis=0))
                rows["Xs"].append(Xs[part].mean(axis=0))
                rows["Y"].append(y_val)
                rows["cycle"].append(int(cyc))
                rows["phase"].append(int(phase_id))
                rows["n_raw"].append(int(part.size))
        result[uid] = {
            "fc": fc,
            "W": np.asarray(rows["W"], dtype=np.float32),
            "Xs": np.asarray(rows["Xs"], dtype=np.float32),
            "Y": np.asarray(rows["Y"], dtype=np.float32),
            "cycle": np.asarray(rows["cycle"], dtype=np.int32),
            "phase": np.asarray(rows["phase"], dtype=np.int16),
            "n_raw": np.asarray(rows["n_raw"], dtype=np.int32),
        }
    return result


def select_validation_units(unit_data: Dict[int, Dict], window: int, val_frac: float) -> tuple[list[int], list[int]]:
    """Choose unit-level validation with FC and RUL coverage."""
    dev_ids = [int(u) for u in sorted(unit_data.keys())]
    eligible = [u for u in dev_ids if len(unit_data[u]["Y"]) >= window]
    if len(eligible) <= 1:
        return dev_ids[:-1], dev_ids[-1:]

    by_fc: dict[int, list[int]] = defaultdict(list)
    for uid in eligible:
        by_fc[int(unit_data[uid]["fc"])].append(uid)

    target_n = max(1, round(len(eligible) * val_frac))
    selected: list[int] = []

    for _fc, uids in sorted(by_fc.items()):
        # Prefer units with broad RUL coverage and enough endpoint windows.
        best = max(
            uids,
            key=lambda u: (
                float(unit_data[u]["Y"].max() - unit_data[u]["Y"].min()),
                len(unit_data[u]["Y"]),
                u,
            ),
        )
        selected.append(best)

    if len(selected) > target_n and len(by_fc) > 1:
        selected = selected[:target_n]

    remaining = [u for u in eligible if u not in selected]
    while len(selected) < target_n and remaining:
        # Add units farthest from already selected max-RUL values.
        selected_max = np.asarray([unit_data[u]["Y"].max() for u in selected], dtype=float)
        nxt = max(
            remaining,
            key=lambda u: float(np.min(np.abs(unit_data[u]["Y"].max() - selected_max))),
        )
        selected.append(nxt)
        remaining.remove(nxt)

    if len(selected) >= len(dev_ids):
        selected = selected[-1:]
    train_ids = [u for u in dev_ids if u not in selected]
    return train_ids, sorted(selected)


def fit_ops_scaler(unit_data: Dict[int, Dict], train_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    W = np.concatenate([unit_data[u]["W"] for u in train_ids], axis=0)
    return W.mean(axis=0), W.std(axis=0) + 1e-8


def one_hot(values: np.ndarray, categories: list[int]) -> np.ndarray:
    idx = {v: i for i, v in enumerate(categories)}
    out = np.zeros((len(values), len(categories)), dtype=np.float32)
    for r, v in enumerate(values.astype(int).tolist()):
        if v in idx:
            out[r, idx[v]] = 1.0
    return out


def design_matrix(
    W_norm: np.ndarray,
    phase: np.ndarray,
    fc: np.ndarray,
    phase_categories: list[int],
    fc_categories: list[int],
) -> np.ndarray:
    intercept = np.ones((len(W_norm), 1), dtype=np.float32)
    phase_oh = one_hot(phase, phase_categories)
    fc_oh = one_hot(fc, fc_categories)
    return np.concatenate([intercept, W_norm.astype(np.float32), phase_oh, fc_oh], axis=1)


def fit_sensor_zscore_only(
    unit_data: Dict[int, Dict],
    train_ids: Iterable[int],
) -> Dict:
    """Null residualizer: global sensor z-score only, no regression subtraction.

    Used when training data has only one flight class so the FC column in the
    design matrix is constant and residualization would produce invalid residuals
    for unseen test FCs.
    """
    Xs = np.concatenate([unit_data[u]["Xs"] for u in train_ids], axis=0)
    phase = np.concatenate([unit_data[u]["phase"] for u in train_ids], axis=0)
    Xs_mean = Xs.mean(axis=0)
    Xs_std = Xs.std(axis=0) + 1e-8
    phase_categories = sorted(np.unique(phase).astype(int).tolist())
    return {
        "mode": "zscore_only",
        "Xs_mean": Xs_mean.astype(np.float32),
        "Xs_std": Xs_std.astype(np.float32),
        "coef": np.zeros((1, Xs.shape[1]), dtype=np.float32),  # no-op subtraction
        "resid_mean": np.zeros(Xs.shape[1], dtype=np.float32),
        "resid_std": np.ones(Xs.shape[1], dtype=np.float32),
        "phase_categories": phase_categories,
        "fc_categories": [],  # no FC in design matrix
    }


def fit_residualizer(
    unit_data: Dict[int, Dict],
    train_ids: Iterable[int],
    W_mean: np.ndarray,
    W_std: np.ndarray,
    ridge: float,
) -> Dict[str, np.ndarray | list[int]]:
    W = np.concatenate([unit_data[u]["W"] for u in train_ids], axis=0)
    Xs = np.concatenate([unit_data[u]["Xs"] for u in train_ids], axis=0)
    phase = np.concatenate([unit_data[u]["phase"] for u in train_ids], axis=0)
    fc = np.concatenate([
        np.full(len(unit_data[u]["Y"]), unit_data[u]["fc"], dtype=np.int16)
        for u in train_ids
    ])

    Xs_mean = Xs.mean(axis=0)
    Xs_std = Xs.std(axis=0) + 1e-8
    Xs_z = (Xs - Xs_mean) / Xs_std
    W_norm = (W - W_mean) / W_std
    phase_categories = sorted(np.unique(phase).astype(int).tolist())
    fc_categories = sorted(np.unique(fc).astype(int).tolist())
    B = design_matrix(W_norm, phase, fc, phase_categories, fc_categories)

    eye = np.eye(B.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    coef = np.linalg.solve(B.T @ B + ridge * eye, B.T @ Xs_z).astype(np.float32)
    resid = Xs_z - B @ coef
    resid_mean = resid.mean(axis=0)
    resid_std = resid.std(axis=0) + 1e-8
    return {
        "Xs_mean": Xs_mean.astype(np.float32),
        "Xs_std": Xs_std.astype(np.float32),
        "coef": coef.astype(np.float32),
        "resid_mean": resid_mean.astype(np.float32),
        "resid_std": resid_std.astype(np.float32),
        "phase_categories": phase_categories,
        "fc_categories": fc_categories,
    }


def apply_transform(
    all_data: Dict[int, Dict],
    W_mean: np.ndarray,
    W_std: np.ndarray,
    residualizer: Dict,
) -> None:
    zscore_only = residualizer.get("mode") == "zscore_only"
    for d in all_data.values():
        W_norm = (d["W"] - W_mean) / W_std
        Xs_z = (d["Xs"] - residualizer["Xs_mean"]) / residualizer["Xs_std"]
        if zscore_only:
            # Single-FC dataset: skip regression subtraction to avoid design-matrix
            # mismatch when test data contains unseen flight classes.
            d["Xs_norm"] = Xs_z.astype(np.float32)
        else:
            fc_arr = np.full(len(d["Y"]), d["fc"], dtype=np.int16)
            B = design_matrix(
                W_norm,
                d["phase"],
                fc_arr,
                list(residualizer["phase_categories"]),
                list(residualizer["fc_categories"]),
            )
            resid = Xs_z - B @ residualizer["coef"]
            resid_z = (resid - residualizer["resid_mean"]) / residualizer["resid_std"]
            d["Xs_norm"] = resid_z.astype(np.float32)
        d["W_norm"] = W_norm.astype(np.float32)


def make_windows(unit_data: Dict[int, Dict], unit_ids: list[int], window: int, stride: int) -> tuple:
    X_list, y_list, uid_list, cycle_list, phase_list, fc_list = [], [], [], [], [], []
    for uid in unit_ids:
        d = unit_data[uid]
        X_unit = np.concatenate([d["W_norm"], d["Xs_norm"]], axis=1)
        n = len(X_unit)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            end = start + window - 1
            X_list.append(X_unit[start:start + window])
            y_list.append(d["Y"][end])
            uid_list.append(uid)
            cycle_list.append(d["cycle"][end])
            phase_list.append(d["phase"][end])
            fc_list.append(d["fc"])
    if not X_list:
        return (
            np.empty((0, window, INPUT_DIM), np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.int64),
            np.empty(0, np.int32),
            np.empty(0, np.int16),
            np.empty(0, np.int16),
        )
    return (
        np.stack(X_list).astype(np.float32),
        np.asarray(y_list, np.float32),
        np.asarray(uid_list, np.int64),
        np.asarray(cycle_list, np.int32),
        np.asarray(phase_list, np.int16),
        np.asarray(fc_list, np.int16),
    )


def summarize_windows(y: np.ndarray, fc: np.ndarray) -> dict:
    out = {
        "n": int(len(y)),
        "rul_min": float(y.min()) if len(y) else None,
        "rul_max": float(y.max()) if len(y) else None,
    }
    if len(fc):
        out["fc_counts"] = {str(int(k)): int((fc == k).sum()) for k in sorted(np.unique(fc))}
    else:
        out["fc_counts"] = {}
    return out


def process_dataset(
    ds_name: str,
    data_dir: Path,
    out_base: Path,
    window: int,
    stride: int,
    n_phase_tokens: int,
    val_frac: float,
    ridge: float,
    verbose: bool = True,
) -> None:
    h5_file = data_dir / DATASETS[ds_name]
    out_dir = out_base / ds_name
    out_dir.mkdir(parents=True, exist_ok=False)
    if verbose:
        print(f"\n[{ds_name}] Loading {h5_file}")
    raw = load_raw(h5_file)

    if verbose:
        print(f"[{ds_name}] Aggregating {n_phase_tokens} phase tokens per flight ...")
    dev_data = aggregate_phase_tokens(raw["A_dev"], raw["W_dev"], raw["X_s_dev"], raw["Y_dev"], n_phase_tokens)
    test_data = aggregate_phase_tokens(raw["A_test"], raw["W_test"], raw["X_s_test"], raw["Y_test"], n_phase_tokens)

    train_ids, val_ids = select_validation_units(dev_data, window=window, val_frac=val_frac)
    test_ids = [int(u) for u in sorted(test_data.keys())]
    all_data = {**dev_data, **test_data}

    W_mean, W_std = fit_ops_scaler(dev_data, train_ids)
    n_fc_train = len({dev_data[u]["fc"] for u in train_ids})
    if n_fc_train == 1:
        # Only one flight class in training → residualization cannot generalise to
        # unseen test FCs (design-matrix FC column is all-zero for those units).
        # Fall back to plain global sensor z-score to preserve cross-FC amplitude
        # information as the discriminative signal.
        residualizer = fit_sensor_zscore_only(dev_data, train_ids)
        if verbose:
            print(f"[{ds_name}] n_fc_train=1 → using sensor z-score only (no residualization)")
    else:
        residualizer = fit_residualizer(dev_data, train_ids, W_mean, W_std, ridge=ridge)
        if verbose:
            print(f"[{ds_name}] n_fc_train={n_fc_train} → using ops+phase+FC residualization")
    apply_transform(all_data, W_mean, W_std, residualizer)

    X_train, y_train, uid_train, cyc_train, phase_train, fc_train_win = make_windows(all_data, train_ids, window, stride)
    X_val, y_val, uid_val, cyc_val, phase_val, fc_val_win = make_windows(all_data, val_ids, window, stride)
    X_test, y_test, uid_test, cyc_test, phase_test, fc_test_win = make_windows(all_data, test_ids, window, stride)

    for name, arr in (
        ("X_train", X_train),
        ("y_train", y_train),
        ("uid_train", uid_train),
        ("cycle_end_train", cyc_train),
        ("phase_end_train", phase_train),
        ("fc_train_window", fc_train_win),
        ("X_val", X_val),
        ("y_val", y_val),
        ("uid_val", uid_val),
        ("cycle_end_val", cyc_val),
        ("phase_end_val", phase_val),
        ("fc_val_window", fc_val_win),
        ("X_test", X_test),
        ("y_test", y_test),
        ("uid_test", uid_test),
        ("cycle_end_test", cyc_test),
        ("phase_end_test", phase_test),
        ("fc_test_window", fc_test_win),
    ):
        np.save(out_dir / f"{name}.npy", arr)

    fc_train_units = {str(u): int(dev_data[u]["fc"]) for u in train_ids}
    fc_val_units = {str(u): int(dev_data[u]["fc"]) for u in val_ids}
    fc_test_units = {str(u): int(test_data[u]["fc"]) for u in test_ids}
    meta = {
        "dataset": ds_name,
        "h5_file": h5_file.name,
        "protocol": "mscs_adapted_phase_tokens_ops_phase_fc_residual",
        "rul_unit": "flight cycles",
        "window": window,
        "stride": stride,
        "n_phase_tokens_per_flight": n_phase_tokens,
        "n_sensors": N_SENSORS,
        "n_ops": N_OPS,
        "input_dim": INPUT_DIM,
        "train_units": train_ids,
        "val_units": val_ids,
        "test_units": test_ids,
        "fc_train": fc_train_units,
        "fc_val": fc_val_units,
        "fc_test": fc_test_units,
        "n_fc_train": len(set(fc_train_units.values())),
        "variable_names": {
            "A": raw["A_var"],
            "W": raw["W_var"],
            "X_s": raw["X_s_var"],
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "window_summary": {
            "train": summarize_windows(y_train, fc_train_win),
            "val": summarize_windows(y_val, fc_val_win),
            "test": summarize_windows(y_test, fc_test_win),
        },
        "normalization": {
            "ops": "global z-score fitted on train units",
            "sensors": (
                "global z-score only (n_fc_train=1: residualization skipped to avoid "
                "design-matrix mismatch on unseen test FCs)"
                if n_fc_train == 1
                else "global z-score then ridge residualization against ops + phase + train-FC, residual z-score"
            ),
            "n_fc_train": n_fc_train,
            "ridge": ridge,
            "W_mean": W_mean.astype(float).tolist(),
            "W_std": W_std.astype(float).tolist(),
            "Xs_mean": residualizer["Xs_mean"].astype(float).tolist(),
            "Xs_std": residualizer["Xs_std"].astype(float).tolist(),
            "resid_mean": residualizer["resid_mean"].astype(float).tolist(),
            "resid_std": residualizer["resid_std"].astype(float).tolist(),
            "phase_categories": list(residualizer["phase_categories"]),
            "fc_categories": list(residualizer["fc_categories"]),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(f"[{ds_name}] train_units={train_ids}")
        print(f"[{ds_name}] val_units={val_ids}")
        print(f"[{ds_name}] test_units={test_ids}")
        print(f"[{ds_name}] shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
        print(f"[{ds_name}] window_summary={meta['window_summary']}")
        print(f"[{ds_name}] Saved -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "raw/ncmapss")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "processed/ncmapss")
    args = parser.parse_args()
    for dataset in args.datasets:
        process_dataset(dataset, args.data_dir, args.output_dir, window=60, stride=1,
                        n_phase_tokens=3, val_frac=0.15, ridge=1e-3)

if __name__ == "__main__":
    main()
