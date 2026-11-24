"""Train a separate final DTMHC model for each requested N-CMAPSS subset and seed."""

import argparse
import json
from pathlib import Path

import torch

from training.engine import train
from training.factory import DATASETS, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=[s for s in DATASETS if s.startswith("DS")],
                        default=[s for s in DATASETS if s.startswith("DS")])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 84, 168])
    parser.add_argument("--data-dir", type=Path, help="Parent of processed N-CMAPSS subset folders")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        for seed in args.seeds:
            cfg = load_config(dataset)
            cfg["seed"] = seed
            if args.dry_run:
                print(json.dumps(cfg))
            else:
                output = args.output_dir / f"{dataset}_s{seed}"
                print(json.dumps(train(cfg, args.data_dir, output, torch.device(args.device))))


if __name__ == "__main__":
    main()
