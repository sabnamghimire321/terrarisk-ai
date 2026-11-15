"""Tests grid generation: cell count, unique IDs, and coordinates within the requested bounds."""

import numpy as np

from src.terrarisk.grid.build_grid import build_grid

TEST_BOUNDS = {
    "min_lat": 26.0,
    "max_lat": 27.0,
    "min_lon": 85.0,
    "max_lon": 86.0,
}
TEST_RESOLUTION = 0.5

def test_grid_has_expected_columns():
    grid = build_grid(bounds=TEST_BOUNDS, resolution_deg=TEST_RESOLUTION)
    assert list(grid.columns) == ["cell_id", "lat", "lon"]

def test_grid_cell_count_matches_bounds_and_resolution():
    grid = build_grid(bounds=TEST_BOUNDS, resolution_deg=TEST_RESOLUTION)

    expected_lats = np.arange(
        TEST_BOUNDS["min_lat"], TEST_BOUNDS["max_lat"], TEST_RESOLUTION
    )
    expected_lons = np.arange(
        TEST_BOUNDS["min_lon"], TEST_BOUNDS["max_lon"], TEST_RESOLUTION
    )
    expected_count = len(expected_lats) * len(expected_lons)

    assert len(grid) == expected_count

def test_grid_cell_ids_are_unique():
    grid = build_grid(bounds=TEST_BOUNDS, resolution_deg=TEST_RESOLUTION)
    assert grid["cell_id"].is_unique

def test_grid_coordinates_within_bounds():
    grid = build_grid(bounds=TEST_BOUNDS, resolution_deg=TEST_RESOLUTION)

    assert (grid["lat"] >= TEST_BOUNDS["min_lat"]).all()
    assert (grid["lat"] < TEST_BOUNDS["max_lat"]).all()
    assert (grid["lon"] >= TEST_BOUNDS["min_lon"]).all()
    assert (grid["lon"] < TEST_BOUNDS["max_lon"]).all()