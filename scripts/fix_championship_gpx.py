"""
fix_championship_gpx.py

World Championships and UEC European Championships GPX files contain only
one lap of the circuit (or partial laps), giving distances ~50-70% shorter
than the real race. This script patches rider CSVs by:
  1. Scaling cumulative features (D+, D-, cobblestones, cols) by the
     ratio real_dist / gpx_dist.
  2. Updating distance_gpx_km to the real distance.
  3. Leaving positional features (gradient_last_Xkm, altitude, denivele_last_5km)
     unchanged — the finish circuit is the same regardless of the number of laps.

Run:
    python scripts/fix_championship_gpx.py
"""

import math
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import causal_model as cm

# Real men's elite distances (km)
REAL_DISTANCES = {
    ('world-championship',                 2017): 267.5,
    ('world-championship',                 2018): 258.5,
    ('world-championship',                 2019): 261.3,
    ('world-championship',                 2020): 258.2,
    ('world-championship',                 2021): 268.3,
    ('world-championship',                 2022): 266.9,
    ('world-championship',                 2023): 271.1,
    ('world-championship',                 2024): 273.9,
    ('world-championship',                 2025): 270.0,
    ('uec-road-european-championships-me', 2018): 188.5,
    ('uec-road-european-championships-me', 2019): 182.3,
    ('uec-road-european-championships-me', 2021): 175.7,
    ('uec-road-european-championships-me', 2022): 182.1,
    ('uec-road-european-championships-me', 2023): 208.0,
}

# Dates needed by find_gpx_path
DATES = {
    ('world-championship',                 2017): '2017-09-24',
    ('world-championship',                 2018): '2018-09-30',
    ('world-championship',                 2019): '2019-09-29',
    ('world-championship',                 2020): '2020-09-27',
    ('world-championship',                 2021): '2021-09-26',
    ('world-championship',                 2022): '2022-09-25',
    ('world-championship',                 2023): '2023-08-06',
    ('world-championship',                 2024): '2024-09-29',
    ('world-championship',                 2025): '2025-09-28',
    ('uec-road-european-championships-me', 2018): '2018-08-12',
    ('uec-road-european-championships-me', 2019): '2019-08-11',
    ('uec-road-european-championships-me', 2021): '2021-09-12',
    ('uec-road-european-championships-me', 2022): '2022-08-14',
    ('uec-road-european-championships-me', 2023): '2023-09-24',
}

# Cumulative features — scale by ratio
CUMUL_FEATS = [
    'denivele_pos', 'denivele_neg',
    'cobblestones_km', 'compacted_gravel_km',
    'cobblestones_last_50km', 'compacted_gravel_last_50km',
    'cobblestones_last_20km', 'compacted_gravel_last_20km',
    'cobblestones_last_10km', 'compacted_gravel_last_10km',
    'n_cols_cat4', 'n_cols_cat3', 'n_cols_cat2', 'n_cols_cat1', 'n_cols_hc',
    'denivele_first_50km',
]

# Positional features — keep unchanged (same finish circuit)
KEEP_FEATS = [
    'altitude_max', 'altitude_min',
    'gradient_last_1km', 'gradient_last_3km', 'gradient_last_5km',
    'denivele_last_5km',
    'loc_last_col_hc', 'loc_last_col_cat1', 'loc_last_col_cat2',
    'loc_last_col_cat3', 'loc_last_col_cat4',
    'gradient_first_50km',
]


def gpx_distance(filepath: str) -> float | None:
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = {'g': 'http://www.topografix.com/GPX/1/1'}
        pts = root.findall('.//g:trkpt', ns)
        lats = [float(p.get('lat')) for p in pts]
        lons = [float(p.get('lon')) for p in pts]
        d = 0.0
        for i in range(1, len(lats)):
            R = 6371
            dlat = math.radians(lats[i] - lats[i-1])
            dlon = math.radians(lons[i] - lons[i-1])
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lats[i-1])) * math.cos(math.radians(lats[i])) * math.sin(dlon/2)**2
            d += R * 2 * math.asin(math.sqrt(max(0, a)))
        return round(d, 2)
    except Exception:
        return None


def build_patch_map() -> dict:
    """Return {(course, year): {'distance_gpx_km': X, 'ratio': R, ...}}"""
    patch = {}
    cm._GPX_INDEX = None
    for (course, year), real_dist in REAL_DISTANCES.items():
        date = DATES.get((course, year))
        if date is None:
            continue
        gpx_path = cm.find_gpx_path(course, date, stage_num=None)
        if gpx_path is None:
            print(f'  ✗ No GPX found for {course} {year}')
            continue
        gpx_dist = gpx_distance(gpx_path)
        if gpx_dist is None or gpx_dist < 10:
            print(f'  ✗ Could not parse GPX for {course} {year}')
            continue
        ratio = real_dist / gpx_dist
        print(f'  {course} {year}: GPX={gpx_dist:.1f}km → real={real_dist:.1f}km  ratio={ratio:.2f}x')
        patch[(course, year)] = {'real_dist': real_dist, 'ratio': ratio}
    return patch


def patch_file(path: Path, patch: dict) -> int:
    df = pd.read_csv(path)
    changed = 0
    for (course, year), info in patch.items():
        mask = (df['course'] == course) & (df['year'] == year)
        if not mask.any():
            continue
        ratio = info['ratio']
        real_dist = info['real_dist']

        # Update distance
        if 'distance_gpx_km' in df.columns:
            df.loc[mask, 'distance_gpx_km'] = round(real_dist, 2)

        # Scale cumulative features
        for feat in CUMUL_FEATS:
            if feat in df.columns:
                df.loc[mask, feat] = (df.loc[mask, feat] * ratio).round(2)

        changed += int(mask.sum())

    if changed:
        df.to_csv(path, index=False)
    return changed


def main():
    print('Building patch map...')
    patch = build_patch_map()
    if not patch:
        print('Nothing to patch.')
        return

    rider_files = sorted(Path('rider_data').glob('*.csv'))
    print(f'\nPatching {len(rider_files)} rider files...')
    files_changed = total_rows = 0
    for f in rider_files:
        n = patch_file(f, patch)
        if n:
            files_changed += 1
            total_rows += n

    print(f'\nDone. Files modified: {files_changed} | Rows patched: {total_rows}')

    # Verification
    print('\nVerification on van_aert_wout.csv:')
    df = pd.read_csv(BASE_DIR / 'rider_data' / 'van_aert_wout.csv')
    mask = df['course'].isin(['world-championship', 'uec-road-european-championships-me'])
    cols = ['course', 'year', 'distance_gpx_km', 'denivele_pos']
    print(df[mask][cols].drop_duplicates(['course', 'year']).sort_values(['course', 'year']).to_string())


if __name__ == '__main__':
    main()
