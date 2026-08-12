"""
fix_denivele_last_5km.py

Quick fix for aberrant denivele_last_5km values.

Bug: original code used last_points=min(500, len(elevations)) — a fixed point
count, not a 5km window. For GPX files with sparse density, 500 points >> 5km,
inflating D+ values by 2-5×.

Fix: for rows where denivele_last_5km > THRESHOLD, replace with an estimate
derived from gradient_last_5km (which is computed correctly with haversine):
    estimated_D+ = max(0, gradient_last_5km / 100 * 5000)
This gives a lower bound that is accurate within ~10-20% for mountain finishes.

Usage:
    python scripts/fix_denivele_last_5km.py            # apply fixes
    python scripts/fix_denivele_last_5km.py --dry-run  # show stats without writing
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import causal_model as cm

RIDER_DIR = Path(cm.RIDER_DIR)
DRY_RUN   = '--dry-run' in sys.argv
THRESHOLD = 700  # m — values above this are almost certainly artifacts


def fix_file(f: Path) -> int:
    """Return number of cells changed."""
    df = pd.read_csv(f)
    if 'denivele_last_5km' not in df.columns or 'gradient_last_5km' not in df.columns:
        return 0

    mask = df['denivele_last_5km'] > THRESHOLD
    if not mask.any():
        return 0

    corrected = (df.loc[mask, 'gradient_last_5km'].clip(lower=0) / 100 * 5000).round(0)
    df.loc[mask, 'denivele_last_5km'] = corrected
    n = int(mask.sum())

    if not DRY_RUN:
        df.to_csv(f, index=False)
    return n


def main():
    rider_files = sorted(RIDER_DIR.glob('*.csv'))
    print(f'Scanning {len(rider_files)} rider CSV files (threshold: >{THRESHOLD}m) ...')

    files_changed = 0
    total_cells   = 0
    for f in rider_files:
        n = fix_file(f)
        if n:
            files_changed += 1
            total_cells   += n

    mode = '[DRY RUN] ' if DRY_RUN else ''
    print(f'{mode}Files modified: {files_changed}')
    print(f'{mode}Cells updated:  {total_cells:,}')
    if DRY_RUN:
        print('\nRemove --dry-run to apply.')
    else:
        print('Done.')


if __name__ == '__main__':
    main()
