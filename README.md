# TerraRisk AI

Code for experimenting with short-horizon flood and landslide risk models for Nepal,
built only on free, open data. It is research code, not a production system and not an
early-warning tool. For real flood or landslide risk, follow official guidance from local
emergency services.

The code covers the whole path: downloading the data, building a feature table, creating
leakage-safe train, validation and test splits, training a LightGBM model, comparing it with
simple baselines, and measuring how uncertain each comparison is.

## Data

| Source | Used for |
|---|---|
| Open-Meteo archive API | Daily precipitation history |
| Open-Meteo Flood API (GloFAS-based) | River discharge history, the basis for flood labels |
| Open-Meteo Elevation API | Static elevation per grid cell |
| NASA Global Landslide Catalog | Landslide events (manual CSV download, ends 2017-07-12) |

Details, URLs and access notes are in `docs/data_sources.md`. No data files are included in
this repository; `data/raw/` and `data/processed/` are empty and are filled by the steps below.
The study area is Nepal on a 0.25-degree grid, with flood and landslide features computed for a
72-cell subsample of it (every third cell in each direction).

## Setup

Requires Python 3.12. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Build the data

Run these from the project root. The downloads need internet access and can take a while; the
flood API in particular throttles long histories.

```powershell
python -m scripts.run_ingestion
python -m src.terrarisk.ingestion.open_meteo_weather_archive
python -m src.terrarisk.ingestion.open_meteo_flood
python -m src.terrarisk.ingestion.landslide_inventory
python -m src.terrarisk.features.build_features
python -m src.terrarisk.baseline.flood_baseline
python -m src.terrarisk.baseline.climatological_baseline
```

The landslide step reads the catalog CSV you export by hand from the NASA viewer into
`data/raw/global_landslide_catalog_raw.csv`; see the docstring in
`src/terrarisk/ingestion/landslide_inventory.py`. After these steps
`data/processed/feature_table.csv` exists, which is what every experiment reads. The last two
commands create baseline files that a few scripts still read for compatibility; they are not
used when the train-only baselines described below are available.

## Run the tests

```powershell
python -m pytest -q
```

The tests use small synthetic data and need no network access or real datasets.

## Run the experiments

Each experiment lives in `experiments/<name>/run.py`, writes a JSON file to
`experiments/<name>/results/`, and has its own tests. Run from the project root.

| Experiment | What it does | Command |
|---|---|---|
| `baseline_ladder` | Baselines, logistic regression, random forest and LightGBM on one split | `python -m experiments.baseline_ladder.run` |
| `calibration` | Class weighting x raw or Platt-scaled probabilities | `python -m experiments.calibration.run` |
| `ablation` | Feature-group ablation | `python -m experiments.ablation.run` |
| `label_sensitivity` | Flood threshold at the 85th, 90th and 95th percentile | `python -m experiments.label_sensitivity.run` |
| `richer_precip_features` | Adds 14-day and 30-day rainfall windows and an anomaly feature | `python -m experiments.richer_precip_features.run --n-bootstrap 1000 --seed 42` |
| `operational_topk` | Events captured in the top 1%, 5% and 10% of scores | `python -m experiments.operational_topk.run` |
| `error_analysis` | Misses and false alarms by latitude, season and rainfall | `python -m experiments.error_analysis.run` |
| `shap_explainability` | Global and local explanations from LightGBM `pred_contrib` | `python -m experiments.shap_explainability.run` |
| `robustness` | Missing data, rainfall noise, forecast horizons | `python -m experiments.robustness.run` |
| `uncertainty` | Paired bootstrap differences against baselines | `python -m experiments.uncertainty.run --n-bootstrap 1000 --seed 42` |
| `rolling_origin` | Five annual test years, flood only | `python -m experiments.rolling_origin.run --n-bootstrap 1000 --seed 42` |
| `spatial_validation` | Held-out latitude block of cells | `python -m experiments.spatial_validation.run --n-bootstrap 1000 --seed 42` |
| `longitude_spatial_validation` | Held-out longitude block of cells | `python -m experiments.longitude_spatial_validation.run --n-bootstrap 1000 --seed 42` |
| `spatiotemporal_validation` | Held-out cells and a held-out final year | `python -m experiments.spatiotemporal_validation.run --n-bootstrap 1000 --seed 42` |
| `elevation_confound_check` | Checks whether an elevation-only model behaves like a per-cell rate | `python -m experiments.elevation_confound_check.run` |

`rolling_origin` also accepts `--weighting selected` for a variant that picks class weighting
on the validation year only. The bootstrap experiments take `--n-bootstrap` and `--seed`; the
defaults are 1000 and 42.

## How the pipeline is built

- **Labels.** A row on date t is positive if the event happens on any of the next seven days
  (t+1 to t+7). Rows without a full seven-day window are dropped, windows never cross grid
  cells, and for the landslide catalog the last usable row is its end date minus seven days.
- **Splits.** Train, then a seven-day gap, then a 365-day validation window, then another
  gap, then a 365-day test window. Every temporal experiment uses the same function
  (`prepare_hazard_data` in `src/terrarisk/models/train_model.py`).
- **Held-out geography.** `src/terrarisk/models/spatial_holdout.py` holds out a block of cells,
  and optionally also the final year, so validation never shares dates with the test period.
- **Baselines.** Built from training rows only: each cell's event rate, and the same rate by
  calendar month. For held-out cells the rate is pooled over the training cells, or over the
  nearest training cells.
- **Model.** One shared LightGBM recipe in `src/terrarisk/models/recipe.py`: early stopping and
  Platt scaling on the validation window. A guard test fails if an experiment builds LightGBM
  any other way.
- **Uncertainty.** `src/terrarisk/models/paired_bootstrap.py` resamples whole grid cells or whole
  14-day blocks of dates, scores the model and a baseline on the same rows, and reports the
  95% interval of the difference.

## Project layout

```
src/terrarisk/
  config.py              paths and constants
  grid/                  the Nepal grid
  ingestion/             one module per data source
  features/              feature table and labels
  baseline/              original baseline scripts
  models/
    train_model.py       labels, splits, train-only baselines, main training entry
    recipe.py            the shared LightGBM recipe
    paired_bootstrap.py  paired cluster bootstrap
    spatial_holdout.py   spatial and spatiotemporal splits
experiments/             one folder per experiment (run.py, tests/, results/)
scripts/run_ingestion.py core data download pipeline
tests/                   unit tests for the shared code
docs/data_sources.md     data sources and access notes
data/                    empty; filled by the data steps above
```

## Results

Each experiment writes its output to `experiments/<name>/results/` as JSON. Those files are
regenerated by running the commands above; the folders are kept in the repository so they have
a place to land.

## License

MIT. The data sources have their own licenses and terms of use; see `docs/data_sources.md`.
