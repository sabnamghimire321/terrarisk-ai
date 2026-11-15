"""
The ONE LightGBM training recipe every experiment should use, so that
numbers from different experiments describe the same model.

Before this module, ten experiment scripts trained 200 trees with no
early stopping, while train_model.py and the three spatial scripts
early-stopped on the validation split. The same nominal "calibrated
LightGBM" therefore produced different PR-AUC values in different
tables. The recipe fixed here:

  - 200 trees max, 15 leaves, learning rate 0.05, seed 42
  - class weighting scale_pos_weight = negatives / positives (train only)
  - early stopping (20 rounds) on the validation split
  - Platt scaling (sigmoid) fit on the validation split only

The test split is never touched by any of these steps.

fit_lightgbm_selected() is a variant used only where an experiment
declares it in advance: it chooses between class weighting on and off
using the validation split ALONE (validation PR-AUC, weighting kept
unless the unweighted model is strictly better). It exists because the
choice must never be made by looking at test results.
"""

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score

LIGHTGBM_PARAMS = {
    "n_estimators": 200,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 20


def fit_lightgbm(X_train, y_train, X_val, y_val, use_class_weight: bool = True):
    """Fits LightGBM on train, early-stopping on the validation split."""
    n_positive = int(y_train.sum())
    n_negative = len(y_train) - n_positive
    scale_pos_weight = (
        n_negative / n_positive if (use_class_weight and n_positive > 0) else 1.0
    )
    model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **LIGHTGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def calibrate_on_validation(model, X_val, y_val):
    """Platt scaling fit on the validation split only (never on test)."""
    calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)
    return calibrator


def fit_lightgbm_selected(X_train, y_train, X_val, y_val):
    """
    Fits LightGBM with class weighting on and off and keeps the one with
    the higher validation PR-AUC. Weighting is kept unless the unweighted
    model is STRICTLY better, so ties and undefined scores keep it.
    Returns (model, used_class_weight, details). Test data is not touched.
    """
    candidates = {}
    for use_weight in (True, False):
        model = fit_lightgbm(X_train, y_train, X_val, y_val, use_class_weight=use_weight)
        validation_pr_auc = float(
            average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        )
        candidates[use_weight] = (model, validation_pr_auc)

    unweighted_is_strictly_better = candidates[False][1] > candidates[True][1]
    chosen = not unweighted_is_strictly_better
    details = {
        "weighted_validation_pr_auc": candidates[True][1],
        "unweighted_validation_pr_auc": candidates[False][1],
    }
    return candidates[chosen][0], chosen, details
