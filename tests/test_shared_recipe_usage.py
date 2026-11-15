"""
Guard test: experiments must train LightGBM through the shared recipe
(src/terrarisk/models/recipe.py), not with their own inline settings.
Different inline settings made "calibrated LightGBM" mean different
models in different tables.

PENDING lists scripts that still build LightGBM inline and are waiting
to be moved onto the recipe. Remove a script from PENDING in the same
change that moves it; the test also fails if a PENDING script no longer
needs to be listed, so the list can only shrink.
"""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"

PENDING: set = set()


def _builds_lightgbm_inline(run_py: Path) -> bool:
    return "LGBMClassifier(" in run_py.read_text(encoding="utf-8")


def test_only_pending_experiments_build_lightgbm_inline():
    offenders = set()
    for run_py in EXPERIMENTS_DIR.glob("*/run.py"):
        if _builds_lightgbm_inline(run_py):
            offenders.add(run_py.parent.name)
    assert offenders == PENDING, (
        f"Experiments building LightGBM inline: {sorted(offenders)}; "
        f"expected exactly the pending set {sorted(PENDING)}."
    )
