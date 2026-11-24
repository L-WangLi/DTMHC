# DTMHC: Dual Timescale Modeling with Historical Context Enhancement for Turbofan Engine Remaining Useful Life Prediction


The final model combines historical segment statistics, a PatchTST recent-window encoder, temporal fusion and RUL regression. It supports C-MAPSS and eight N-CMAPSS subsets, with a separately trained model for each subset.

## Setup and data

Use Python 3.10 and install the dependencies:

```bash
pip install -r requirements.txt
```

Obtain the datasets from the [NASA data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

- Place C-MAPSS `train_FD00*.txt`, `test_FD00*.txt` and `RUL_FD00*.txt` in `dataset/raw/cmapss/`. Preprocessing runs during training or evaluation.
- Place N-CMAPSS `.h5` files in `dataset/raw/ncmapss/`, then run:

```bash
python -m dataset.preprocess_ncmapss
```

N-CMAPSS arrays are saved under `dataset/processed/ncmapss/<subset>/`. Use `--datasets DS01` to process one subset; `--data-dir` and `--output-dir` select other locations.

## Training and evaluation

Run from the repository root:

```bash
python train.py --dataset FD001 --seed 7
python train.py --dataset DS01 --seed 7
python run_ncmapss.py --datasets DS01 DS04 --seeds 7 21 42 84 168
python train.py --dataset FD001 --dry-run

python evaluate.py --checkpoint runs/FD001_s7/best.pt \
  --output-dir outputs/FD001_s7
```

`--data-dir` points to raw C-MAPSS text files or the parent of processed N-CMAPSS subset folders. `--device cpu` selects CPU execution. Training saves its best validation checkpoint, configuration, logs and test results under `runs/<dataset>_s<seed>/`. Existing output directories are never overwritten.

Evaluation reports RMSE and NASA Score using each C-MAPSS test engine's last observed endpoint or all full-window interval endpoints for N-CMAPSS. Add `--all-prefixes` for evaluation over every observed prefix. Checkpoints contain their configuration and engine-split protocol.

## Code and configuration

- `models/dtmhc.py`: final model; `models/layers.py`: PatchTST and attention layers.
- `dataset/`: preprocessing, windows, masks and historical statistics.
- `training/`: model/data construction, training and evaluation.
- `configs/cmapss/` and `configs/ncmapss/`: final hyperparameters and preprocessing settings.
- `configs/protocols.json`: centralized engine splits, expressed with inclusive ranges.

Training uses RUL/125 with MSE; reported predictions are `125 * clip(output, 0, 1)`. C-MAPSS labels are capped at 125; N-CMAPSS labels are uncapped flight-cycle RUL. All preprocessing is fitted on training engines only. N-CMAPSS uses three equal-sample intervals per flight; DS08c uses sensor z-scores without residualization.

 Citation information is in `CITATION.cff`.
