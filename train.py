"""Train the final DTMHC model on one C-MAPSS or N-CMAPSS subset."""

import argparse
import json
from pathlib import Path

import torch

from training.engine import train
from training.factory import DATASETS, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="FD001")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", type=Path,
                        help="Raw C-MAPSS folder, or parent of processed N-CMAPSS subset folders")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print settings without reading data or training")
    args = parser.parse_args()
    cfg = load_config(args.dataset)
    cfg["seed"] = args.seed
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    output = args.output_dir or Path("runs") / f"{args.dataset}_s{args.seed}"
    print(json.dumps(train(cfg, args.data_dir, output, torch.device(args.device)), indent=2))


if __name__ == "__main__":
    main()
