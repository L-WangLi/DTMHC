"""Final supervised training and evaluation protocol for C-MAPSS and N-CMAPSS."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset.common import gather_windows, test_window_index, train_window_index
from dataset.history import history_tokens
from .factory import build_model, load_data


def build_inputs(data, trajs, index, cfg):
    x, pad = gather_windows(trajs, index, data.window)
    hx, hpad, hoff = history_tokens(trajs, index, data.window, cfg["hist_seg"],
                                   cfg["hist_slots"], "x_hist" if cfg["hist_condres"] else "x")
    y = np.array([trajs[k].rul[e] for k, e in index], dtype=np.float32)
    arrays = dict(x=x, pad=pad, hx=hx, hpad=hpad, hoff=hoff, y=y)
    meta = dict(unit=np.array([trajs[k].unit for k, _ in index]),
                condition=np.array([trajs[k].cond[e] for k, e in index]),
                end=np.asarray(index)[:, 1], true=y)
    return {k: torch.as_tensor(v) for k, v in arrays.items()}, meta


def place(split, device, limit=3e9):
    size = sum(v.numel() * v.element_size() for v in split.values())
    if device.type == "cuda" and size < limit:
        return {k: v.to(device) for k, v in split.items()}
    return {k: v.pin_memory() if device.type == "cuda" else v for k, v in split.items()}


def batch(split, idx, device):
    return {k: v[idx.to(v.device)].to(device, non_blocking=True) for k, v in split.items()}


@torch.no_grad()
def predict(model, split, device, cfg):
    model.eval()
    ys = []
    for i in range(0, len(split["y"]), cfg["eval_batch_size"]):
        idx = torch.arange(i, min(i + cfg["eval_batch_size"], len(split["y"])))
        ys.append(model(batch(split, idx, device))["y"].float())
    return (torch.cat(ys).clamp(0, 1) * cfg["target_scale"]).cpu().numpy()


def metrics(pred, true):
    error = pred - true
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                score=float(np.sum(np.where(error < 0, np.exp(-error / 13.0) - 1,
                                            np.exp(error / 10.0) - 1))),
                n_predictions=len(true))


def write_predictions(path, pred, meta):
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["unit", "cycle_index", "condition", "true", "pred"])
        for i, value in enumerate(pred):
            w.writerow([int(meta["unit"][i]), int(meta["end"][i]), int(meta["condition"][i]),
                        float(meta["true"][i]), float(value)])


def test_index(data, all_prefixes=False):
    if all_prefixes:
        return train_window_index(data.test, data.window, short_prefixes=True)
    if data.name.startswith("DS"):
        return train_window_index(data.test, data.window)
    return test_window_index(data.test)


def evaluate_model(model, data, cfg, device, output_dir, all_prefixes=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    split, meta = build_inputs(data, data.test, test_index(data, all_prefixes), cfg)
    pred = predict(model, place(split, device), device, cfg)
    result = metrics(pred, meta["true"])
    result.update(dataset=data.name, all_prefixes=all_prefixes,
                  evaluation="all observed prefixes" if all_prefixes else (
                      "all full-window interval endpoints" if data.name.startswith("DS")
                      else "last observed endpoint per engine"))
    write_predictions(output_dir / "predictions.csv", pred, meta)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def train(cfg, data_dir, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = load_data(cfg, data_dir)
    model = build_model(cfg).to(device)
    tr, _ = build_inputs(data, data.train, train_window_index(
        data.train, data.window, cfg["train_short_prefixes"]), cfg)
    va, vm = build_inputs(data, data.val, train_window_index(
        data.val, data.window, cfg["train_short_prefixes"]), cfg)
    tr, va = place(tr, device), place(va, device)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["eta_min"])
    best, best_epoch, bad, log = float("inf"), -1, 0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(tr["y"]))
        total_loss = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            opt.zero_grad(set_to_none=True)
            b = batch(tr, idx, device)
            out = model(b)
            loss = F.mse_loss(out["y"].float(), b["y"] / cfg["target_scale"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            total_loss += float(loss.detach()) * len(idx)
        sched.step()
        val_rmse = metrics(predict(model, va, device, cfg), vm["true"])["rmse"]
        log.append(dict(epoch=epoch, loss_rul=total_loss / len(perm), val_rmse=val_rmse))
        print(json.dumps(log[-1]), flush=True)
        if val_rmse < best - cfg["min_delta"]:
            best, best_epoch, bad = val_rmse, epoch, 0
            torch.save(dict(state_dict=model.state_dict(), config=cfg, epoch=epoch,
                            val_rmse=best), output_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    with (output_dir / "train_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss_rul", "val_rmse"], lineterminator="\n")
        w.writeheader()
        w.writerows(log)
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    result = evaluate_model(model, data, cfg, device, output_dir / "test")
    result.update(best_epoch=best_epoch, val_rmse=best, epochs_run=len(log))
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
