"""
fix_more_missing_gpx.py

Patches rider CSVs for (course, year) pairs where distance_gpx_km == 0.
Handles:
  - One-day races: compute features directly from GPX
  - Stage races: compute features per stage GPX, match by (course, year, date, stage_num)
  - Scaling cases (women's GPX): scale cumulative features by real_dist / gpx_dist

Run:
    python fix_more_missing_gpx.py
"""

import xml.etree.ElementTree as ET
import math
import numpy as np
import pandas as pd
from pathlib import Path

import causal_model as cm

BASE_DIR = Path(__file__).parent

# ── One-day races ────────────────────────────────────────────────────────────
# (course, year, date)
ONEDAY_TARGETS = [
    ('gooikse-pijl',                2023, '2023-09-17'),
    ('gooikse-pijl',                2024, '2024-09-22'),
    ('gp-de-fourmies',              2018, '2018-09-02'),
    ('gp-de-fourmies',              2019, '2019-09-08'),
    ('gp-de-fourmies',              2021, '2021-09-12'),
    ('gp-de-fourmies',              2022, '2022-09-11'),
    ('gp-de-fourmies',              2023, '2023-09-10'),
    ('gp-de-fourmies',              2024, '2024-09-08'),
    ('gp-de-fourmies',              2025, '2025-09-14'),
    ('memorial-frank-vandenbroucke', 2021, '2021-10-05'),
    ('memorial-frank-vandenbroucke', 2023, '2023-10-03'),
    ('memorial-frank-vandenbroucke', 2024, '2024-10-01'),
    ('memorial-frank-vandenbroucke', 2025, '2025-10-07'),
    ('memorial-rik-van-steenbergen', 2022, '2022-10-09'),
    ('ronde-van-limburg',           2021, '2021-05-24'),
    ('schaal-schels',               2022, '2022-08-21'),
]

# ── Stage races ───────────────────────────────────────────────────────────────
# (course, year, [(date, stage_num), ...])
STAGE_TARGETS = [
    ('4-jours-de-dunkerque', 2018, [
        ('2018-05-08', 1), ('2018-05-09', 2), ('2018-05-10', 3),
        ('2018-05-11', 4), ('2018-05-12', 5), ('2018-05-13', 6),
    ]),
    ('4-jours-de-dunkerque', 2019, [
        ('2019-05-14', 1), ('2019-05-15', 2), ('2019-05-16', 3),
        ('2019-05-17', 4), ('2019-05-18', 5), ('2019-05-19', 6),
    ]),
    ('4-jours-de-dunkerque', 2022, [
        ('2022-05-03', 1), ('2022-05-04', 2), ('2022-05-05', 3),
        ('2022-05-06', 4), ('2022-05-07', 5), ('2022-05-08', 6),
    ]),
    ('okolo-jiznich-cech', 2023, [
        ('2023-09-07', 1), ('2023-09-08', 2), ('2023-09-09', 3), ('2023-09-10', 4),
    ]),
]

# ── Scaling targets (women's GPX — cumulative features scaled) ────────────────
# (course, year, date, real_dist_km)
SCALING_TARGETS = [
    ('tre-valli-varesine', 2024, '2024-10-08', 197.3),
]

CUMUL_FEATS = [
    'denivele_pos', 'denivele_neg',
    'cobblestones_km', 'compacted_gravel_km',
    'cobblestones_last_50km', 'compacted_gravel_last_50km',
    'cobblestones_last_20km', 'compacted_gravel_last_20km',
    'cobblestones_last_10km', 'compacted_gravel_last_10km',
    'n_cols_cat4', 'n_cols_cat3', 'n_cols_cat2', 'n_cols_cat1', 'n_cols_hc',
    'denivele_first_50km',
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(max(0, a)))


def parse_gpx(filepath):
    try:
        tree = ET.parse(filepath)
    except ET.ParseError as e:
        print(f'    ✗ XML parse error in {Path(filepath).name}: {e}')
        return None
    root = tree.getroot()
    ns = {'g': 'http://www.topografix.com/GPX/1/1'}
    pts = root.findall('.//g:trkpt', ns)
    if not pts:
        return None
    lats = [float(p.get('lat')) for p in pts]
    lons = [float(p.get('lon')) for p in pts]
    ele_nodes = [p.find('g:ele', ns) for p in pts]
    eles = [float(e.text) if e is not None else None for e in ele_nodes]

    dists = [0.0]
    for i in range(1, len(lats)):
        dists.append(dists[-1] + haversine_km(lats[i-1], lons[i-1], lats[i], lons[i]))

    df = pd.DataFrame({'dist': dists, 'ele': eles}).dropna()
    return df


def smooth_ele(ele, window=5):
    kernel = np.ones(window) / window
    return np.convolve(ele, kernel, mode='same')


def compute_features(gpx_df):
    dist = gpx_df['dist'].values
    ele = smooth_ele(gpx_df['ele'].values)

    total_dist = float(dist[-1])
    diff = np.diff(ele)
    deniv_pos = float(diff[diff > 0].sum())
    deniv_neg = float(abs(diff[diff < 0].sum()))

    def gradient_last_x(x_km):
        idx = np.searchsorted(dist, dist[-1] - x_km)
        idx = max(0, min(idx, len(dist) - 2))
        d = dist[-1] - dist[idx]
        return float((ele[-1] - ele[idx]) / (d * 10)) if d >= 0.01 else 0.0

    def deniv_last_x(x_km):
        idx = np.searchsorted(dist, dist[-1] - x_km)
        idx = max(0, min(idx, len(dist) - 2))
        seg = np.diff(ele[idx:])
        return float(seg[seg > 0].sum())

    def gradient_first_x(x_km):
        idx = np.searchsorted(dist, x_km)
        idx = max(1, min(idx, len(dist) - 1))
        d = dist[idx] - dist[0]
        return float((ele[idx] - ele[0]) / (d * 10)) if d >= 0.01 else 0.0

    def deniv_first_x(x_km):
        idx = np.searchsorted(dist, x_km)
        idx = max(1, min(idx, len(dist) - 1))
        seg = np.diff(ele[:idx+1])
        return float(seg[seg > 0].sum())

    return {
        'distance_gpx_km':    round(total_dist, 2),
        'denivele_pos':        round(deniv_pos, 1),
        'denivele_neg':        round(deniv_neg, 1),
        'altitude_max':        round(float(ele.max()), 1),
        'altitude_min':        round(float(ele.min()), 1),
        'gradient_last_1km':   round(gradient_last_x(1), 2),
        'gradient_last_3km':   round(gradient_last_x(3), 2),
        'gradient_last_5km':   round(gradient_last_x(5), 2),
        'denivele_last_5km':   round(deniv_last_x(5), 1),
        'gradient_first_50km': round(gradient_first_x(50), 2),
        'denivele_first_50km': round(deniv_first_x(50), 1),
    }


def build_patch_map():
    """
    Returns two dicts:
      direct_patch  : {(course, year, date, stage_num_or_None) -> feats}
      scaling_patch : {(course, year) -> {real_dist, ratio}}
    """
    cm._GPX_INDEX = None
    direct_patch = {}
    scaling_patch = {}

    # One-day races
    for course, year, date in ONEDAY_TARGETS:
        gpx_path = cm.find_gpx_path(course, date, stage_num=None)
        if gpx_path is None:
            print(f'  ✗ {course} {year}: no GPX')
            continue
        gpx_df = parse_gpx(gpx_path)
        if gpx_df is None or len(gpx_df) < 10:
            print(f'  ✗ {course} {year}: parse failed')
            continue
        feats = compute_features(gpx_df)
        direct_patch[(course, year, date, None)] = feats
        print(f'  ✓ {course} {year}: {feats["distance_gpx_km"]}km  D+={feats["denivele_pos"]}m')

    # Stage races
    for course, year, stages in STAGE_TARGETS:
        seen_gpx = {}
        for date, stage_num in stages:
            gpx_path = cm.find_gpx_path(course, date, stage_num=stage_num)
            if gpx_path is None:
                print(f'  ✗ {course} {year} stage {stage_num}: no GPX')
                continue
            if gpx_path not in seen_gpx:
                gpx_df = parse_gpx(gpx_path)
                if gpx_df is None or len(gpx_df) < 10:
                    print(f'  ✗ {course} {year} stage {stage_num}: parse failed')
                    continue
                seen_gpx[gpx_path] = compute_features(gpx_df)
            feats = seen_gpx[gpx_path]
            direct_patch[(course, year, date, stage_num)] = feats
            print(f'  ✓ {course} {year} stage {stage_num}: {feats["distance_gpx_km"]}km')

    # Scaling targets
    for course, year, date, real_dist in SCALING_TARGETS:
        gpx_path = cm.find_gpx_path(course, date, stage_num=None)
        if gpx_path is None:
            print(f'  ✗ {course} {year}: no GPX for scaling')
            continue
        gpx_df = parse_gpx(gpx_path)
        if gpx_df is None or len(gpx_df) < 10:
            print(f'  ✗ {course} {year}: parse failed for scaling')
            continue
        feats = compute_features(gpx_df)
        gpx_dist = feats['distance_gpx_km']
        ratio = real_dist / gpx_dist
        scaling_patch[(course, year)] = {'real_dist': real_dist, 'ratio': ratio, 'feats': feats}
        print(f'  ✓ {course} {year} [scale]: GPX={gpx_dist:.1f}km → real={real_dist:.1f}km ratio={ratio:.2f}x')

    return direct_patch, scaling_patch


def patch_file(path: Path, direct_patch: dict, scaling_patch: dict) -> int:
    df = pd.read_csv(path)
    changed = 0

    # Direct patch: match by (course, year, date, stage_num)
    for (course, year, date, stage_num), feats in direct_patch.items():
        if stage_num is None:
            mask = ((df['course'] == course) & (df['year'] == year)
                    & (df['date'] == date) & (df['distance_gpx_km'] == 0))
        else:
            mask = ((df['course'] == course) & (df['year'] == year)
                    & (df['date'] == date) & (df['stage_num'] == stage_num)
                    & (df['distance_gpx_km'] == 0))
        if not mask.any():
            continue
        for col, val in feats.items():
            if col in df.columns:
                df.loc[mask, col] = val
        changed += int(mask.sum())

    # Scaling patch: match by (course, year)
    for (course, year), info in scaling_patch.items():
        mask = (df['course'] == course) & (df['year'] == year) & (df['distance_gpx_km'] == 0)
        if not mask.any():
            continue
        ratio = info['ratio']
        feats = info['feats']

        if 'distance_gpx_km' in df.columns:
            df.loc[mask, 'distance_gpx_km'] = round(info['real_dist'], 2)

        # Scale cumulative features
        for feat in CUMUL_FEATS:
            if feat in df.columns:
                df.loc[mask, feat] = (df.loc[mask, feat] * ratio).round(2)

        # Apply positional features from GPX directly
        for col in ['altitude_max', 'altitude_min', 'gradient_last_1km',
                    'gradient_last_3km', 'gradient_last_5km', 'denivele_last_5km',
                    'gradient_first_50km']:
            if col in df.columns and col in feats:
                df.loc[mask, col] = feats[col]

        changed += int(mask.sum())

    if changed:
        df.to_csv(path, index=False)
    return changed


def main():
    print('Building GPX patch map...')
    direct_patch, scaling_patch = build_patch_map()
    total_entries = len(direct_patch) + len(scaling_patch)
    if total_entries == 0:
        print('Nothing to patch.')
        return

    rider_files = sorted(Path('rider_data').glob('*.csv'))
    print(f'\nPatching {len(rider_files)} rider CSV files...')
    files_changed = total_rows = 0
    for f in rider_files:
        n = patch_file(f, direct_patch, scaling_patch)
        if n:
            files_changed += 1
            total_rows += n

    print(f'\nDone. Files modified: {files_changed} | Rows patched: {total_rows}')

    # Verification on Dillier
    print('\nVerification — dillier_silvan.csv remaining zeros:')
    df = pd.read_csv(BASE_DIR / 'rider_data' / 'dillier_silvan.csv')
    zeros = df[df['distance_gpx_km'] == 0][['course', 'year', 'stage_num', 'date']]
    if zeros.empty:
        print('  ✓ No zeros remaining!')
    else:
        print(zeros.to_string())


if __name__ == '__main__':
    main()
