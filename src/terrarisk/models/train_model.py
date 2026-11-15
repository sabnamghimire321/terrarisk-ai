"""
Trains a LightGBM classifier per hazard (flood, landslide) and
evaluates it against that hazard's climatological baseline.

LEAKAGE: river_discharge_m3s / flood_threshold_m3s are excluded from
features (flood_event is literally defined from them).

FORECAST HORIZON: labels are shifted forward - a row's label is
"does an event occur in the next H days," using only day-t-and-earlier
features. Not a same-day nowcast mirror.

Two real leakage bugs were found and fixed during a formal audit
(not invented in advance):

1. Label window crossing the train/val boundary. A row's horizon
   label looks up to HORIZON_DAYS days into the future, so a row near
   the end of the training period had a label depending on events in
   the validation period. Fixed by drop_boundary_leakage_rows(),
   which drops the last HORIZON_DAYS days of what would otherwise be
   the training set.
2. Test-window sizing computed on data that included the test period
   itself. choose_test_window_days() must only ever see a train-only
   slice, or the evaluation protocol's own design would be informed
   by test-period event rates.

3. Validation labels reaching into the test period. The same forward
   look-ahead exists at the validation/test boundary, and validation
   labels drive early stopping and Platt scaling. Fixed by purging the
   last HORIZON_DAYS days of validation as well (time_based_split's
   purge_days).
4. Incomplete label windows. pandas max() skips missing values, so
   rows near the end of a cell's series used to get a label from fewer
   than HORIZON_DAYS future days. add_horizon_label() now returns NaN
   unless the full window exists, and prepare_hazard_data() drops those
   rows.

5. Baselines using test-period outcomes. The CSV baselines in
   data/processed are computed from the whole history (and, for
   landslide, from datetime.now()), so they contain validation and test
   outcomes. prepare_hazard_data() now attaches baselines computed from
   the TRAIN rows only (compute_train_only_baselines) to the test
   frame, and get_baseline_predictions() uses them.

prepare_hazard_data() is the single source of truth for the
leakage-safe split - every experiment in experiments/ reuses it
rather than reimplementing splitting logic.
"""

import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss

from src.terrarisk.config import DATA_PROCESSED
from src.terrarisk.grid.build_grid import build_grid

FEATURE_TABLE_PATH = DATA_PROCESSED / "feature_table.csv"

FEATURE_COLUMNS = [
    "precip_1day_sum_mm",
    "precip_3day_sum_mm",
    "precip_7day_sum_mm",
    "elevation_m",
    "day_of_year",
]

HORIZON_DAYS = 7
MIN_EXPECTED_TEST_POSITIVES = 15

PROTOCOL_MIN_TEST_DAYS = 365
MIN_TEST_DAYS_DEFAULT = PROTOCOL_MIN_TEST_DAYS

TRAIN_ONLY_BASELINE_COL = "train_only_baseline"
TRAIN_ONLY_SEASONAL_BASELINE_COL = "train_only_seasonal_baseline"
POOLED_BASELINE_EVENT_COLUMNS = {"landslide_event"}
BASELINE_POOL_SIZE = 15
SEASONAL_SHRINKAGE_ROWS = 30


def add_horizon_label(
    df: pd.DataFrame, event_col: str, horizon_days: int, new_col_name: str
) -> pd.DataFrame:
    """
    Shifts a same-day event flag into a forward-looking horizon label.

    A row on date t gets label 1 if the event occurs on any of the days
    t+1 ... t+horizon_days (the current day t is NOT part of the
    window), 0 if it occurs on none of them, and NaN if the full window
    is not available. The window is NOT available when:
      - fewer than horizon_days later rows exist for that cell,
      - any date inside the window is missing for that cell, or
      - the event flag itself is missing inside the window.
    Callers must drop NaN labels. Windows never cross cell boundaries.
    Raises ValueError on duplicate (cell_id, date) rows, because the
    correct label for a duplicated date is ambiguous.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    df = df.sort_values(["cell_id", "date"]).copy()

    if df.duplicated(subset=["cell_id", "date"]).any():
        raise ValueError(
            "Duplicate (cell_id, date) rows found; de-duplicate before "
            "building horizon labels."
        )

    dates = pd.to_datetime(df["date"])
    by_cell_dates = dates.groupby(df["cell_id"], sort=False)

    last_window_date = by_cell_dates.shift(-horizon_days)
    window_is_complete = (last_window_date - dates) == pd.Timedelta(days=horizon_days)

    by_cell_events = df.groupby("cell_id", sort=False)[event_col]
    future_events = pd.concat(
        [by_cell_events.shift(-h) for h in range(1, horizon_days + 1)], axis=1
    )

    label = future_events.max(axis=1, skipna=False)
    df[new_col_name] = label.where(window_is_complete)

    return df

def choose_test_window_days(
    df: pd.DataFrame, label_col: str, min_expected_positives: int
) -> int:
    """
    Sizes the test window to expect at least `min_expected_positives`
    events, based on this hazard's own positive rate within `df`
    (already scoped to a train-only slice by the caller - never the
    full usable range, or the sizing itself would be informed by
    test-period data).
    """
    overall_rate = df[label_col].mean()
    total_days_available = (df["date"].max() - df["date"].min()).days
    rows_per_day = len(df) / max(total_days_available, 1)

    if overall_rate <= 0:
        return 0

    days_needed = min_expected_positives / (overall_rate * rows_per_day)
    return int(np.ceil(days_needed))

def time_based_split(
    df: pd.DataFrame,
    test_days: int,
    val_days: int,
    max_usable_date=None,
    purge_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Forward-chaining split by global date cutoffs - never random.

    The test period is exactly `test_days` calendar days ending on the
    anchor date (inclusive). The validation period is the `val_days`
    days before it. Everything earlier is train.

    `purge_days` should equal the label horizon. A row on date d has a
    label that looks at d+1 ... d+purge_days, so the last `purge_days`
    days before each boundary are dropped from the earlier period:

        TRAIN | purge | VALIDATION | purge | TEST

    This keeps every label window of an earlier period strictly before
    the first date of the next period. With purge_days=0 nothing is
    dropped (kept so older callers and tests behave as before).

    `max_usable_date` anchors the split to the last date that is
    actually meaningful for this hazard's labels.
    """
    anchor_date = max_usable_date if max_usable_date is not None else df["date"].max()
    purge = pd.Timedelta(days=purge_days)

    test_start = anchor_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[df["date"] < val_start - purge]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start - purge)]
    test = df[(df["date"] >= test_start) & (df["date"] <= anchor_date)]

    return train, val, test

def drop_boundary_leakage_rows(
    train: pd.DataFrame, horizon_days: int
) -> pd.DataFrame:
    """
    Drops the last `horizon_days` days of `train`. Without this, a
    row near the end of the training period has a label computed from
    events up to `horizon_days` days after the train/val cutoff - the
    row's features are legitimately train-period, but its label
    indirectly encodes validation-period information. This creates a
    real gap with no label overlap across the train/val boundary.
    """
    if len(train) == 0:
        return train
    cutoff = train["date"].max() - pd.Timedelta(days=horizon_days)
    return train[train["date"] <= cutoff]


def _pools_for_hazard(event_col: str, cell_ids) -> dict:
    """
    {cell_id: [cell_ids pooled into its baseline]}. Singleton pools for
    dense hazards (flood). For sparse hazards (landslide) each cell is
    pooled with its BASELINE_POOL_SIZE nearest cells in this table
    (Euclidean distance on lat/lon, ties broken by grid order so the
    result is deterministic).
    """
    cell_ids = list(cell_ids)
    if event_col not in POOLED_BASELINE_EVENT_COLUMNS:
        return {c: [c] for c in cell_ids}

    grid = build_grid()
    grid = grid[grid["cell_id"].isin(cell_ids)].reset_index(drop=True)
    missing = set(cell_ids) - set(grid["cell_id"])
    if missing:
        raise ValueError(
            f"Cannot pool baseline: {len(missing)} cell ids are not on the "
            f"project grid (e.g. {sorted(missing)[:3]})."
        )

    lat = grid["lat"].to_numpy()
    lon = grid["lon"].to_numpy()
    distances = np.sqrt((lat[:, None] - lat[None, :]) ** 2
                        + (lon[:, None] - lon[None, :]) ** 2)
    ids = grid["cell_id"].to_numpy()
    k = min(BASELINE_POOL_SIZE, len(ids))
    return {
        ids[i]: ids[np.argsort(distances[i], kind="stable")[:k]].tolist()
        for i in range(len(ids))
    }


def compute_train_only_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str,
    pools: dict = None,
    shrinkage_rows: int = SEASONAL_SHRINKAGE_ROWS,
) -> pd.DataFrame:
    """
    Climatological baselines estimated ONLY from `train` rows, for every
    row of `test`. Returns a DataFrame indexed like `test` with two columns:

      train_only_baseline            - the cell's (pooled) event rate over
                                       all training rows. Same definition
                                       as the label, so Brier is comparable.
      train_only_seasonal_baseline   - the same rate for the row's calendar
                                       month, shrunk toward the all-year
                                       rate: (events + k*rate) / (rows + k).
                                       A fair opponent for a model that
                                       knows the day of year.

    `pools` maps cell_id -> list of cell ids pooled into that cell's
    estimate (default: each cell alone). Cells with no training data fall
    back to the overall training rate. Test labels are never read.
    """
    if len(train) == 0:
        raise ValueError("Cannot compute baselines from an empty training set.")

    overall_rate = float(train[label_col].mean())

    flat = pd.DataFrame({
        "cell_id": train["cell_id"].to_numpy(),
        "month": pd.to_datetime(train["date"]).dt.month.to_numpy(),
        "y": train[label_col].to_numpy(),
    })
    grouped = flat.groupby(["cell_id", "month"])["y"]
    sums = grouped.sum().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    counts = grouped.count().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)

    test_cells = list(pd.unique(test["cell_id"]))
    cell_rate = {}
    seasonal_rate = {}
    for cell in test_cells:
        members = (pools or {}).get(cell, [cell])
        members = [m for m in members if m in sums.index]
        if not members:
            cell_rate[cell] = overall_rate
            for month in range(1, 13):
                seasonal_rate[(cell, month)] = overall_rate
            continue
        pooled_sum = sums.loc[members].sum(axis=0)
        pooled_count = counts.loc[members].sum(axis=0)
        rate = float(pooled_sum.sum() / pooled_count.sum())
        cell_rate[cell] = rate
        for month in range(1, 13):
            denominator = pooled_count[month] + shrinkage_rows
            seasonal_rate[(cell, month)] = (
                float((pooled_sum[month] + shrinkage_rows * rate) / denominator)
                if denominator > 0 else rate
            )

    test_months = pd.to_datetime(test["date"]).dt.month.to_numpy()
    out = pd.DataFrame(index=test.index)
    out[TRAIN_ONLY_BASELINE_COL] = test["cell_id"].map(cell_rate).fillna(overall_rate).to_numpy()
    out[TRAIN_ONLY_SEASONAL_BASELINE_COL] = [
        seasonal_rate.get((c, m), overall_rate)
        for c, m in zip(test["cell_id"].to_numpy(), test_months)
    ]
    return out


def prepare_hazard_data(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    max_usable_date=None,
    min_test_days: int = MIN_TEST_DAYS_DEFAULT,
    horizon_days: int = HORIZON_DAYS,
    verbose: bool = True,
    protocol_floor_days: int = PROTOCOL_MIN_TEST_DAYS,
):
    """
    Single source of truth for turning the raw feature table into a
    leakage-safe (train, val, test, label_col) tuple for one hazard.
    Used by every model in this project, from the climatological
    baseline through LightGBM, so the leakage fixes described in
    this module's docstring only ever have to live in one place.

    Returns None if this hazard cannot be reliably evaluated at all
    with current data (e.g. not enough usable date range) - callers
    must check for this and skip gracefully, not assume success.

    `min_test_days` can never go below `protocol_floor_days` (default
    365, one full seasonal cycle). Experiments that still pass an older
    value such as 60 are therefore raised to the floor here, in the one
    shared place. Unit tests on tiny synthetic data pass
    protocol_floor_days=0 explicitly.
    """

    min_test_days = max(min_test_days, protocol_floor_days)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"HAZARD: {hazard_name} (horizon={horizon_days} days)")
        print(f"{'=' * 60}")

    label_col = f"{event_col}_horizon_label"
    labeled = add_horizon_label(df, event_col, horizon_days, label_col)
    labeled = labeled.dropna(subset=[label_col])
    labeled[label_col] = labeled[label_col].astype(int)

    if max_usable_date is not None:
        anchor_date = pd.Timestamp(max_usable_date) - pd.Timedelta(days=horizon_days)
        if verbose:
            print(f"Event source ends {pd.Timestamp(max_usable_date).date()}; "
                  f"last row with a fully observed {horizon_days}-day label "
                  f"window is {anchor_date.date()}. Split is anchored there "
                  f"(weather range ends {labeled['date'].max().date()}).")
        usable = labeled[labeled["date"] <= anchor_date]
    else:
        anchor_date = labeled["date"].max()
        usable = labeled

    usable_span_days = (usable["date"].max() - usable["date"].min()).days
    sizing_cutoff = usable["date"].min() + pd.Timedelta(days=int(usable_span_days * 0.7))
    sizing_slice = usable[usable["date"] <= sizing_cutoff]

    test_days = choose_test_window_days(sizing_slice, label_col, MIN_EXPECTED_TEST_POSITIVES)
    test_days = max(test_days, min_test_days)
    val_days = test_days

    total_usable_days = (usable["date"].max() - usable["date"].min()).days
    if (test_days + val_days + 2 * horizon_days) >= total_usable_days:
        if verbose:
            print(f"CANNOT reliably evaluate {hazard_name}: a window meeting "
                  f"both the >= {MIN_EXPECTED_TEST_POSITIVES}-positives target "
                  f"and the {min_test_days}-day minimum span would need "
                  f"{test_days}+{val_days} days (plus two {horizon_days}-day "
                  f"purge gaps), but only {total_usable_days} "
                  f"usable days exist. Real data-volume limitation, not a bug.")
        return None

    if verbose:
        print(f"Test window: {test_days} days (>= {MIN_EXPECTED_TEST_POSITIVES} "
              f"expected positives AND >= {min_test_days}-day minimum span, "
              f"sized using only the first 70% of usable data).")

    train, val, test = time_based_split(
        usable, test_days, val_days, anchor_date, purge_days=horizon_days
    )

    pools = _pools_for_hazard(event_col, usable["cell_id"].unique())
    baselines = compute_train_only_baselines(train, test, label_col, pools)
    test = test.copy()
    for column in baselines.columns:
        test[column] = baselines[column].to_numpy()

    if verbose:
        print(f"Attached train-only baselines to the test frame "
              f"(columns {list(baselines.columns)}).")
        print(f"Purged the last {horizon_days} days of train and of validation "
              f"so no label window crosses into the next period.")
        print(f"Train: {len(train)} rows, Val: {len(val)} rows, Test: {len(test)} rows")
        print(f"Positive counts — train: {int(train[label_col].sum())}, "
              f"val: {int(val[label_col].sum())}, "
              f"test: {int(test[label_col].sum())}")

    if train[label_col].sum() == 0:
        if verbose:
            print(f"WARNING: zero positive examples in training data. Skipping.")
        return None
    if test[label_col].sum() == 0 and verbose:
        print(f"WARNING: test window still has zero positives — metrics "
              f"computed from it will NOT be trustworthy.")

    return train, val, test, label_col

def get_baseline_predictions(
    test: pd.DataFrame, baseline_df: pd.DataFrame = None
) -> pd.Series:
    """
    Climatological baseline probability for each test row.

    If `test` came from prepare_hazard_data(), it carries a baseline
    computed from the training rows only, and that is returned (the
    `baseline_df` argument is then ignored). Otherwise the legacy CSV
    baseline is used, which was computed from the whole history and may
    contain validation/test outcomes; a warning says so. Cells missing
    from baseline_df fall back to its mean rather than producing NaN.
    """
    if TRAIN_ONLY_BASELINE_COL in test.columns:
        return test[TRAIN_ONLY_BASELINE_COL].copy()

    if baseline_df is None:
        raise ValueError(
            "test has no train-only baseline column and no baseline_df was given."
        )
    warnings.warn(
        "Using a full-history baseline CSV: it may include validation/test "
        "outcomes. Prefer a test frame from prepare_hazard_data().",
        UserWarning,
        stacklevel=2,
    )
    baseline_lookup = baseline_df.set_index("cell_id")["baseline_probability"]
    return test["cell_id"].map(baseline_lookup).fillna(baseline_lookup.mean())


def train_and_evaluate(
    df: pd.DataFrame,
    event_col: str,
    hazard_name: str,
    baseline_df: pd.DataFrame,
    max_usable_date=None,
    min_test_days: int = MIN_TEST_DAYS_DEFAULT,
) -> None:
    prepared = prepare_hazard_data(
        df, event_col, hazard_name, max_usable_date, min_test_days
    )
    if prepared is None:
        return
    train, val, test, label_col = prepared

    X_train, y_train = train[FEATURE_COLUMNS], train[label_col]
    X_val, y_val = val[FEATURE_COLUMNS], val[label_col]
    X_test, y_test = test[FEATURE_COLUMNS], test[label_col]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    test_predictions_raw = model.predict_proba(X_test)[:, 1]
    model_pr_auc_raw = average_precision_score(y_test, test_predictions_raw)
    model_brier_raw = brier_score_loss(y_test, test_predictions_raw)

    calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)

    test_predictions_calibrated = calibrator.predict_proba(X_test)[:, 1]
    model_pr_auc_calibrated = average_precision_score(y_test, test_predictions_calibrated)
    model_brier_calibrated = brier_score_loss(y_test, test_predictions_calibrated)

    baseline_predictions = get_baseline_predictions(test, baseline_df)
    baseline_pr_auc = average_precision_score(y_test, baseline_predictions)
    baseline_brier = brier_score_loss(y_test, baseline_predictions)

    print(f"\n--- Results on held-out test period ---")
    print(f"{'Metric':<22}{'Baseline':>13}{'LightGBM raw':>15}{'LightGBM calib.':>17}")
    print(f"{'PR-AUC':<22}{baseline_pr_auc:>13.4f}{model_pr_auc_raw:>15.4f}"
          f"{model_pr_auc_calibrated:>17.4f}")
    print(f"{'Brier (lower=better)':<22}{baseline_brier:>13.4f}{model_brier_raw:>15.4f}"
          f"{model_brier_calibrated:>17.4f}")

    if model_brier_calibrated < baseline_brier:
        print(f"\nCalibrated LightGBM beats the baseline on Brier score for {hazard_name}.")
    else:
        print(f"\nCalibrated LightGBM does NOT beat the baseline on Brier score for {hazard_name}.")

    if model_pr_auc_calibrated > baseline_pr_auc:
        print(f"LightGBM beats the climatological baseline on PR-AUC for {hazard_name}.")
    else:
        print(f"LightGBM does NOT beat the climatological baseline on PR-AUC for {hazard_name}.")

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\nFeature importance:\n{feature_importance.to_string(index=False)}")


if __name__ == "__main__":
    print("Loading feature table...")
    df = pd.read_csv(FEATURE_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    flood_baseline = pd.read_csv(DATA_PROCESSED / "flood_baseline.csv")
    flood_baseline_7d = flood_baseline[flood_baseline["horizon_days"] == HORIZON_DAYS]

    landslide_baseline = pd.read_csv(DATA_PROCESSED / "landslide_baseline.csv")
    landslide_baseline_7d = landslide_baseline[
        landslide_baseline["horizon_days"] == HORIZON_DAYS
    ]

    train_and_evaluate(df, "flood_event", "FLOOD", flood_baseline_7d)

    landslide_anchor = pd.Timestamp("2017-07-12")
    train_and_evaluate(
        df, "landslide_event", "LANDSLIDE", landslide_baseline_7d,
        max_usable_date=landslide_anchor,
        min_test_days=365,
    )