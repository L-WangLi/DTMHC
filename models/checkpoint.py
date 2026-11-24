"""Load checkpoints produced by the DTMHC training entry point."""

from pathlib import Path

import torch


def load_checkpoint(path, device="cpu"):
    from training.factory import build_model

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if "state_dict" not in payload or "config" not in payload:
        raise ValueError("Expected a best.pt checkpoint produced by train.py")
    cfg = payload["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), cfg
