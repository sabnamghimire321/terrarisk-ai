"""
Leakage-safe splits for SPATIAL and SPATIOTEMPORAL holdout experiments:
the test set is a block of grid cells the model never sees in training or
validation.

spatial_holdout_split
    TEST       all labelled rows of the held-out cells (their whole history)
    VALIDATION the last `val_days` days of the OTHER cells
    TRAIN      the other cells, strictly earlier than validation minus the
               label horizon (a purge gap)

spatiotemporal_holdout_split  (unseen geography AND unseen time)
    TEST       held-out cells, the final `test_days` days
    VALIDATION the OTHER cells, the `val_days` days BEFORE the test period
               (minus a purge gap), so validation never shares a date with
               test: events on the same days in neighbouring cells cannot
               reach early stopping or calibration
    TRAIN      the other cells, strictly earlier than validation minus a
               purge gap

Baselines for held-out cells come from the TRAINING cells only; a cell the
model has never seen cannot have its own history used (the old full-history
baseline CSV did exactly that):
  - pooled:    the event rate of all training cells, and by calendar month
  - neighbour: the same, from the `neighbour_pool_size` nearest training
               cells by latitude/longitude (only when a grid is supplied).
               This is the stronger, fairer opponent: it carries geography.

Rows only get a label when the full look-ahead window exists
(add_horizon_label), and for an event source with a known end date
(max_usable_date, the landslide catalog) the split is anchored at
max_usable_date minus the horizon.

Design note for the purely spatial split: the test cells' dates overlap the
training cells' dates, which is inherent in a spatial holdout (one storm hits
cells inside and outside the block on the same days). State it as a
limitation next to any spatial-only result.
"""

import numpy as np
import pandas as pd

from src.terrarisk.models.train_model import (
    HORIZON_DAYS,
    TRAIN_ONLY_BASELINE_COL,
    TRAIN_ONLY_SEASONAL_BASELINE_COL,
    add_horizon_label,
    compute_train_only_baselines,
)

VAL_DAYS_DEFAULT = 365
TEST_DAYS_DEFAULT = 365
NEIGHBOUR_POOL_SIZE = 15

NEIGHBOUR_BASELINE_COL = "train_only_neighbour_baseline"
NEIGHBOUR_SEASONAL_BASELINE_COL = "train_only_neighbour_seasonal_baseline"


def cells_in_bands(grid: pd.DataFrame, column: str, values) -> set:
    """cell_ids whose `column` (e.g. "lat" or "lon") is one of `values`."""
    return set(grid.loc[grid[column].isin(values), "cell_id"])


def nearest_training_cells(
    grid: pd.DataFrame, held_cells, training_cells, k: int = NEIGHBOUR_POOL_SIZE
) -> dict:
    """
    {held cell: its k nearest training cells} by Euclidean distance on
    lat/lon (the project's convention); ties break by sorted cell id, so the
    result is deterministic. `grid` needs columns cell_id, lat, lon.
    """
    coordinates = grid.drop_duplicates("cell_id").set_index("cell_id")[["lat", "lon"]]
    needed = list(held_cells) + list(training_cells)
    missing = [cell for cell in needed if cell not in coordinates.index]
    if missing:
        raise ValueError(
            f"{len(missing)} cells are missing from the grid (e.g. {missing[:3]})."
        )

    training_ids = sorted(training_cells)
    training_xy = coordinates.loc[training_ids].to_numpy(dtype=float)
    k = min(k, len(training_ids))
    pools = {}
    for cell in held_cells:
        held_xy = coordinates.loc[cell].to_numpy(dtype=float)
        distances = np.sqrt(((training_xy - held_xy) ** 2).sum(axis=1))
        order = np.argsort(distances, kind="stable")[:k]
        pools[cell] = [training_ids[i] for i in order]
    return pools


def _attach_baselines(train, test, label_col, grid=None, neighbour_pool_size=NEIGHBOUR_POOL_SIZE):
    training_cells = sorted(train["cell_id"].unique())
    held_cells = list(test["cell_id"].unique())

    pooled_pools = {cell: training_cells for cell in held_cells}
    pooled = compute_train_only_baselines(train, test, label_col, pooled_pools)

    test = test.copy()
    test[TRAIN_ONLY_BASELINE_COL] = pooled[TRAIN_ONLY_BASELINE_COL].to_numpy()
    test[TRAIN_ONLY_SEASONAL_BASELINE_COL] = pooled[TRAIN_ONLY_SEASONAL_BASELINE_COL].to_numpy()

    if grid is not None:
        neighbour_pools = nearest_training_cells(
            grid, held_cells, training_cells, neighbour_pool_size
        )
        neighbour = compute_train_only_baselines(train, test, label_col, neighbour_pools)
        test[NEIGHBOUR_BASELINE_COL] = neighbour[TRAIN_ONLY_BASELINE_COL].to_numpy()
        test[NEIGHBOUR_SEASONAL_BASELINE_COL] = neighbour[TRAIN_ONLY_SEASONAL_BASELINE_COL].to_numpy()
    return test


def _labelled_rows(df, event_col, horizon_days, max_usable_date):
    label_col = f"{event_col}_horizon_label"
    labeled = add_horizon_label(df, event_col, horizon_days, label_col)
    labeled = labeled.dropna(subset=[label_col])
    labeled[label_col] = labeled[label_col].astype(int)

    if max_usable_date is not None:
        anchor = pd.Timestamp(max_usable_date) - pd.Timedelta(days=horizon_days)
    else:
        anchor = labeled["date"].max()
    return labeled[labeled["date"] <= anchor], label_col, anchor


def spatial_holdout_split(
    df: pd.DataFrame,
    event_col: str,
    holdout_cell_ids,
    horizon_days: int = HORIZON_DAYS,
    max_usable_date=None,
    val_days: int = VAL_DAYS_DEFAULT,
    grid: pd.DataFrame = None,
    neighbour_pool_size: int = NEIGHBOUR_POOL_SIZE,
):
    """
    Returns (train, val, test, label_col) with train-only baselines attached
    to `test`, or None when the data cannot support the split.
    """
    holdout = set(holdout_cell_ids)
    usable, label_col, anchor = _labelled_rows(df, event_col, horizon_days, max_usable_date)

    in_holdout = usable["cell_id"].isin(holdout)
    test = usable[in_holdout]
    remaining = usable[~in_holdout]
    if test.empty or remaining.empty:
        return None

    val_start = anchor - pd.Timedelta(days=val_days - 1)
    purge = pd.Timedelta(days=horizon_days)
    val = remaining[remaining["date"] >= val_start]
    train = remaining[remaining["date"] < val_start - purge]
    if train.empty or val.empty or train[label_col].sum() == 0:
        return None

    test = _attach_baselines(train, test, label_col, grid, neighbour_pool_size)
    return train, val, test, label_col


def spatiotemporal_holdout_split(
    df: pd.DataFrame,
    event_col: str,
    holdout_cell_ids,
    horizon_days: int = HORIZON_DAYS,
    max_usable_date=None,
    val_days: int = VAL_DAYS_DEFAULT,
    test_days: int = TEST_DAYS_DEFAULT,
    grid: pd.DataFrame = None,
    neighbour_pool_size: int = NEIGHBOUR_POOL_SIZE,
):
    """
    Unseen geography AND unseen time. Returns (train, val, test, label_col)
    or None. Validation (other cells) and test (held-out cells) never share
    a date, and every label window of an earlier period ends before the next
    period starts.
    """
    holdout = set(holdout_cell_ids)
    usable, label_col, anchor = _labelled_rows(df, event_col, horizon_days, max_usable_date)
    purge = pd.Timedelta(days=horizon_days)

    test_start = anchor - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    in_holdout = usable["cell_id"].isin(holdout)
    test = usable[in_holdout & (usable["date"] >= test_start)]
    others = usable[~in_holdout]
    val = others[(others["date"] >= val_start) & (others["date"] < test_start - purge)]
    train = others[others["date"] < val_start - purge]

    if test.empty or val.empty or train.empty or train[label_col].sum() == 0:
        return None

    test = _attach_baselines(train, test, label_col, grid, neighbour_pool_size)
    return train, val, test, label_col
