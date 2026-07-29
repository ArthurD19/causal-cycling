"""
fix_missing_gpx_features.py

Patches rider CSVs for 8 (course, year) pairs where distance_gpx_km == 0
because the GPX file existed but the matching failed at build time.

Affected races:
  - dwars-door-vlaanderen  2018, 2022
  - antwerp-port-epic      2020, 2021, 2022, 2023, 2024, 2025

Run:
    python fix_missing_gpx_features.py
"""

import xml.etree.ElementTree as ET
import math
import numpy as np
import pandas as pd
from pathlib import Path

import causal_model as cm

BASE_DIR = Path(__file__).parent

TARGETS = [
    ('dwars-door-vlaanderen', 2018, '2018-03-28'),
    ('dwars-door-vlaanderen', 2022, '2022-03-30'),
    ('antwerp-port-epic',     2020, '2020-09-13'),
    ('antwerp-port-epic',     2021, '2021-09-12'),
    ('antwerp-port-epic',     2022, '2022-05-22'),
    ('antwerp-port-epic',     2023, '2023-05-21'),
    ('antwerp-port-epic',     2024, '2024-05-19'),
    ('antwerp-port-epic',     2025, '2025-06-09'),
]

GPX_FEATS = [
    'distance_gpx_km', 'denivele_pos', 'denivele_neg',
    'altitude_max', 'altitude_min',
    'gradient_last_1km', 'gradient_last_3km', 'gradient_last_5km',
    'denivele_last_5km', 'gradient_first_50km', 'denivele_first_50km',
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, a)))


def parse_gpx(filepath):
    """Parse GPX → arrays of (distance_km, elevation)."""
    tree = ET.parse(filepath)
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


def smooth_ele(ele: np.ndarray, window: int = 5) -> np.ndarray:
    """Light moving-average smoothing to remove GPS noise."""
    kernel = np.ones(window) / window
    return np.convolve(ele, kernel, mode='same')


def compute_features(gpx_df: pd.DataFrame) -> dict:
    dist = gpx_df['dist'].values
    ele  = smooth_ele(gpx_df['ele'].values)

    total_dist = float(dist[-1])
    alt_max    = float(ele.max())
    alt_min    = float(ele.min())

    # D+ / D-
    diff = np.diff(ele)
    deniv_pos = float(diff[diff > 0].sum())
    deniv_neg = float(abs(diff[diff < 0].sum()))

    def gradient_last_x(x_km):
        """Average gradient (%) over the last x km."""
        idx = np.searchsorted(dist, dist[-1] - x_km)
        idx = max(0, min(idx, len(dist) - 2))
        d = dist[-1] - dist[idx]
        if d < 0.01:
            return 0.0
        return float((ele[-1] - ele[idx]) / (d * 10))  # % = m/km / 10

    def deniv_last_x(x_km):
        """Total positive elevation gain over last x km."""
        idx = np.searchsorted(dist, dist[-1] - x_km)
        idx = max(0, min(idx, len(dist) - 2))
        seg_diff = np.diff(ele[idx:])
        return float(seg_diff[seg_diff > 0].sum())

    def gradient_first_x(x_km):
        """Average gradient (%) over the first x km."""
        idx = np.searchsorted(dist, x_km)
        idx = max(1, min(idx, len(dist) - 1))
        d = dist[idx] - dist[0]
        if d < 0.01:
            return 0.0
        return float((ele[idx] - ele[0]) / (d * 10))

    def deniv_first_x(x_km):
        """Total positive elevation gain over first x km."""
        idx = np.searchsorted(dist, x_km)
        idx = max(1, min(idx, len(dist) - 1))
        seg_diff = np.diff(ele[:idx+1])
        return float(seg_diff[seg_diff > 0].sum())

    return {
        'distance_gpx_km':    round(total_dist, 2),
        'denivele_pos':        round(deniv_pos, 1),
        'denivele_neg':        round(deniv_neg, 1),
        'altitude_max':        round(alt_max, 1),
        'altitude_min':        round(alt_min, 1),
        'gradient_last_1km':   round(gradient_last_x(1), 2),
        'gradient_last_3km':   round(gradient_last_x(3), 2),
        'gradient_last_5km':   round(gradient_last_x(5), 2),
        'denivele_last_5km':   round(deniv_last_x(5), 1),
        'gradient_first_50km': round(gradient_first_x(50), 2),
        'denivele_first_50km': round(deniv_first_x(50), 1),
    }


def build_patch_map():
    """Return dict: (course, year) → feature dict."""
    patch = {}
    cm._GPX_INDEX = None  # force fresh index
    for course, year, date in TARGETS:
        gpx_path = cm.find_gpx_path(course, date, stage_num=None)
        if gpx_path is None:
            print(f'  ✗ No GPX found for {course} {year}')
            continue
        print(f'  ✓ {course} {year} → {Path(gpx_path).name}')
        gpx_df = parse_gpx(gpx_path)
        if gpx_df is None or len(gpx_df) < 10:
            print(f'    ✗ Could not parse GPX')
            continue
        feats = compute_features(gpx_df)
        print(f'    dist={feats["distance_gpx_km"]} km  D+={feats["denivele_pos"]} m')
        patch[(course, year)] = feats
    return patch


def patch_file(path: Path, patch: dict) -> int:
    df = pd.read_csv(path)
    changed = 0
    for (course, year), feats in patch.items():
        mask = (df['course'] == course) & (df['year'] == year) & (df['distance_gpx_km'] == 0)
        if not mask.any():
            continue
        for col, val in feats.items():
            if col in df.columns:
                df.loc[mask, col] = val
        changed += int(mask.sum())
    if changed:
        df.to_csv(path, index=False)
    return changed


def main():
    print('Building GPX patch map...')
    patch = build_patch_map()
    if not patch:
        print('Nothing to patch.')
        return

    rider_files = sorted(Path('rider_data').glob('*.csv'))
    print(f'\nPatching {len(rider_files)} rider CSV files...')
    files_changed = total_rows = 0
    for f in rider_files:
        n = patch_file(f, patch)
        if n:
            files_changed += 1
            total_rows += n

    print(f'\nDone. Files modified: {files_changed} | Rows patched: {total_rows}')


if __name__ == '__main__':
    main()
