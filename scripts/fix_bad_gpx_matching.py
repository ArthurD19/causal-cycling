"""
fix_bad_gpx_matching.py

Five (course, year) pairs have a GPX file matched to the wrong course:
the terrain data (distance, D+, cols) belongs to a different race, making
these ITT courses appear as mountain road stages.

Fix: null out all GPX-derived columns for the affected rows.
With zeroed terrain features the KMeans will naturally assign them to the
CLM cluster (shortest distance / lowest D+), which is correct for ITTs.
"""

import glob
import pandas as pd
from pathlib import Path

BAD_MATCHINGS = {
    ('nc-ukraine-itt',                2019),
    ('asian-cycling-championships-itt', 2022),
    ('nc-dominican-republic-itt',     2019),
    ('nc-china-itt',                  2019),
    ('nc-china-itt',                  2020),
}

GPX_COLS = [
    'distance_gpx_km',
    'asphalt_km', 'paved_km', 'cobblestones_km', 'unpaved_km',
    'compacted_gravel_km', 'road_km', 'street_km', 'state_road_km',
    'cycleway_km', 'off-grid_(unknown)_km', 'access_road_km', 'unknown_km',
    'path_km', 'singletrack_km',
    'denivele_pos', 'denivele_neg', 'altitude_max', 'altitude_min',
    'n_cols_cat4', 'n_cols_cat3', 'n_cols_cat2', 'n_cols_cat1', 'n_cols_hc',
    'loc_last_col_cat4', 'loc_last_col_cat3', 'loc_last_col_cat2',
    'loc_last_col_cat1', 'loc_last_col_hc',
    'gradient_last_1km', 'gradient_last_3km', 'gradient_last_5km',
    'denivele_last_5km', 'gradient_first_50km', 'denivele_first_50km',
    'cobblestones_last_50km', 'compacted_gravel_last_50km',
    'cobblestones_last_20km', 'compacted_gravel_last_20km',
    'cobblestones_last_10km', 'compacted_gravel_last_10km',
]


def fix_file(path: Path) -> int:
    df = pd.read_csv(path)
    mask = pd.Series(False, index=df.index)
    for course, year in BAD_MATCHINGS:
        mask |= (df['course'] == course) & (df['year'] == year)
    if not mask.any():
        return 0
    cols_present = [c for c in GPX_COLS if c in df.columns]
    df.loc[mask, cols_present] = float('nan')
    df.to_csv(path, index=False)
    return int(mask.sum())


def main():
    rider_files = sorted(Path('rider_data').glob('*.csv'))
    print(f'Scanning {len(rider_files)} rider CSV files...')
    files_changed = total_rows = 0
    for f in rider_files:
        n = fix_file(f)
        if n:
            files_changed += 1
            total_rows += n
            print(f'  {f.name}: {n} row(s) fixed')
    print(f'\nDone. Files modified: {files_changed} | Rows nulled: {total_rows}')


if __name__ == '__main__':
    main()
