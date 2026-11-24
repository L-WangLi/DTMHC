"""Evaluate a final DTMHC checkpoint on the original dataset-level test protocol."""

import argparse
import json
from pathlib import Path

import torch

from models.checkpoint import load_checkpoint
from training.engine import evaluate_model
from training.factory import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path,
                        help="Raw C-MAPSS folder, or parent of processed N-CMAPSS subset folders")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--all-prefixes", action="store_true",
                        help="Evaluate every observed prefix, including padded short windows")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cfg = load_checkpoint(args.checkpoint, device)
    data = load_data(cfg, args.data_dir)
    result = evaluate_model(model, data, cfg, device, args.output_dir, args.all_prefixes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
