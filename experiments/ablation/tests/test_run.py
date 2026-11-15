"""Tests build_feature_combinations() - the 7 feature-group combinations."""

from experiments.ablation.run import FEATURE_GROUPS, build_feature_combinations

def test_seven_combinations_are_produced():
    combos = build_feature_combinations()
    assert len(combos) == 7

def test_all_features_combination_includes_every_column():
    combos = build_feature_combinations()
    all_columns = {c for group in FEATURE_GROUPS.values() for c in group}
    assert set(combos["all_features"]) == all_columns

def test_group_only_combinations_contain_exactly_that_groups_columns():
    combos = build_feature_combinations()
    for group_name, columns in FEATURE_GROUPS.items():
        assert set(combos[f"{group_name}_only"]) == set(columns)

def test_leave_one_out_combinations_exclude_exactly_one_group():
    combos = build_feature_combinations()
    all_columns = {c for group in FEATURE_GROUPS.values() for c in group}

    for excluded_group_name, excluded_columns in FEATURE_GROUPS.items():
        combo = combos[f"without_{excluded_group_name}"]
        assert set(combo) == all_columns - set(excluded_columns)

def test_no_combination_has_duplicate_columns():
    combos = build_feature_combinations()
    for combo_name, columns in combos.items():
        assert len(columns) == len(set(columns))