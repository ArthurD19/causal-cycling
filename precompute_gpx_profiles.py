"""
Precompute elevation profiles for all GPX files.

Reads every .gpx in gpx_files_2/ and gpx_files_ele2/, extracts
(distance_km, elevation), and writes a single gpx_profiles.parquet.

Run once:
    python precompute_gpx_profiles.py

The resulting file is used by load_gpx_profile() instead of parsing
raw GPX files at runtime — reducing the required data from ~11 GB to ~30 MB.
"""

import xml.etree.ElementTree as ET
import math
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
GPX_DIRS = [
    BASE_DIR / 'data' / 'gpx_files_ele2',
    BASE_DIR / 'data' / 'gpx_files_2',
]
OUT_PATH  = BASE_DIR / 'gpx_profiles.parquet'
MAX_PTS   = 600


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def parse_gpx(filepath: Path, max_points: int = MAX_PTS):
    try:
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
            dists.append(dists[-1] + _haversine_km(lats[i-1], lons[i-1], lats[i], lons[i]))

        df = pd.DataFrame({'distance_km': dists, 'elevation': eles}).dropna()
        if len(df) > max_points:
            step = max(1, len(df) // max_points)
            df = df.iloc[::step].reset_index(drop=True)
        df['distance_km'] = df['distance_km'].astype('float32')
        df['elevation']   = df['elevation'].astype('float32')
        return df
    except Exception:
        return None


def main():
    all_gpx = []
    for gpx_dir in GPX_DIRS:
        if gpx_dir.exists():
            all_gpx.extend(sorted(gpx_dir.glob('*.gpx')))

    print(f"Found {len(all_gpx)} GPX files. Parsing…")

    chunks = []
    errors = 0
    for i, path in enumerate(all_gpx):
        if i % 500 == 0:
            print(f"  {i}/{len(all_gpx)}…")
        df = parse_gpx(path)
        if df is None or len(df) == 0:
            errors += 1
            continue
        df.insert(0, 'gpx_key', path.stem)
        chunks.append(df)

    if not chunks:
        print("No profiles extracted.")
        return

    result = pd.concat(chunks, ignore_index=True)
    result['gpx_key'] = result['gpx_key'].astype('category')
    result.to_parquet(OUT_PATH, index=False, compression='snappy')

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nDone. {len(chunks)} profiles ({errors} errors).")
    print(f"Output: {OUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
